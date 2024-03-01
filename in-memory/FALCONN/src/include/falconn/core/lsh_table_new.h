#ifndef __LSH_TABLE_NEW_H__
#define __LSH_TABLE_NEW_H__

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <string>

#include "../falconn_global.h"
#include "data_storage.h"
#include "lsh_table.h"
#include "lsh_query_new.h"

namespace falconn {
namespace core {


template <typename PointType,  // the type of the data points to be stored
          typename KeyType,    // must be integral for a static table
          typename LSH,        // the LSH family
          typename HashType,   // type returned by a set of k LSH functions
          typename HashTable,  // the low-level hash tables
          typename MultiProbe,
          typename DataStorageType = ArrayDataStorage<PointType, KeyType>>
class StaticLSHTable2
    : public BasicLSHTable<LSH, HashTable,
                           StaticLSHTable2<PointType, KeyType, LSH, HashType,
                                          HashTable, MultiProbe, DataStorageType> > {
 public:

 typedef typename LSH::QueryType QueryType;
  StaticLSHTable2(LSH* lsh, HashTable* hash_table, const DataStorageType& points,
                 int_fast32_t num_setup_threads, bool save_index, std::string filename)
      : BasicLSHTable<LSH, HashTable,
                      StaticLSHTable2<PointType, KeyType, LSH, HashType,
                                     HashTable, MultiProbe, DataStorageType>>(lsh,
                                                                  hash_table),
        n_(points.size()),
        points_(points) {
    if (num_setup_threads < 0) {
      throw LSHTableError("Number of setup threads cannot be negative.");
    }
    if (num_setup_threads == 0) {
      num_setup_threads = std::max(1u, std::thread::hardware_concurrency());
    }
    int_fast32_t l = this->lsh_->get_l();

    num_setup_threads = std::min(l, num_setup_threads);
    int_fast32_t num_tables_per_thread = l / num_setup_threads;
    int_fast32_t num_leftover_tables = l % num_setup_threads;

    std::vector<std::future<void>> thread_results;
    int_fast32_t next_table_range_start = 0;

    for (int_fast32_t ii = 0; ii < num_setup_threads; ++ii) {
      int_fast32_t next_table_range_end =
          next_table_range_start + num_tables_per_thread - 1;
      if (ii < num_leftover_tables) {
        next_table_range_end += 1;
      }
      thread_results.push_back(std::async(
          std::launch::async, &StaticLSHTable2::setup_table_range, this,
          next_table_range_start, next_table_range_end, points, save_index, filename));
      next_table_range_start = next_table_range_end + 1;
    }

    for (int_fast32_t ii = 0; ii < num_setup_threads; ++ii) {
      thread_results[ii].get();
    }
  }

  void add_table() {
    typename LSH::template BatchHash<DataStorageType> bh(*(this->lsh_));
    std::vector<HashType> table_hashes;
    bh.batch_hash_single_table(points_, (this->lsh_)->get_l() - 1,
                               &table_hashes);
    this->hash_table_->add_entries_for_table(table_hashes,
                                             (this->lsh_)->get_l() - 1);
  }

  // TODO: add query statistics back in
  class Query {
   public:
    Query(const StaticLSHTable2& parent, unsigned num_probes)
        : parent_(parent),
          is_candidate_(parent.n_),
          lsh_query_(*(parent.lsh_), num_probes) {}

    void get_candidates_with_duplicates(const QueryType& p,
                                        int_fast64_t num_probes,
                                        int_fast64_t max_num_candidates,
                                        std::vector<KeyType>* result,
                                        std::vector<int>* candidate_num = nullptr) {
      if (result == nullptr) {
        throw LSHTableError("Results vector pointer is nullptr.");
      }

      stats_.num_queries += 1;

      auto start_time = std::chrono::high_resolution_clock::now();

      lsh_query_.get_transformed_vector(p);

      auto lsh_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_lsh =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              lsh_end_time - start_time);
      stats_.average_lsh_time += elapsed_lsh.count();

      lsh_query_.get_probes_by_table(&tmp_probes_by_table_, num_probes);

      auto multiprobe_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_multiprobe =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              multiprobe_end_time - lsh_end_time);
      stats_.average_multiprobe_time += elapsed_multiprobe.count();

      hash_table_iterators_ =
          parent_.hash_table_->retrieve_bulk(tmp_probes_by_table_);

      int_fast64_t num_candidates = 0;
      result->clear();
      if (max_num_candidates < 0) {
        max_num_candidates = std::numeric_limits<int_fast64_t>::max();
      }
      while (num_candidates < max_num_candidates &&
             hash_table_iterators_.first != hash_table_iterators_.second) {
        num_candidates += 1;
        result->push_back(*(hash_table_iterators_.first));
        ++hash_table_iterators_.first;
      }
      if (candidate_num != nullptr) {
        candidate_num->operator=(std::move(hash_table_iterators_.first.candidate_nums()));
      }

      auto hashing_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_hashing =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              hashing_end_time - lsh_end_time);
      stats_.average_hash_table_time += elapsed_hashing.count();

      stats_.average_num_candidates += num_candidates;

      auto end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_total =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                    start_time);
      stats_.average_total_query_time += elapsed_total.count();
    }

    void get_unique_candidates(const QueryType& p, int_fast64_t num_probes,
                               int_fast64_t max_num_candidates,
                               std::vector<KeyType>* result) {
      if (result == nullptr) {
        throw LSHTableError("Results vector pointer is nullptr.");
      }

      auto start_time = std::chrono::high_resolution_clock::now();
      stats_.num_queries += 1;

      get_unique_candidates_internal(p, num_probes, max_num_candidates, result);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_total =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                    start_time);
      stats_.average_total_query_time += elapsed_total.count();
    }

    void get_unique_candidates_and_tables(const QueryType& p, int_fast64_t num_probes,
                               int_fast64_t max_num_candidates,
                               std::vector<FalconnCandidateType>* result) {
      if (result == nullptr) {
        throw LSHTableError("Results vector pointer is nullptr.");
      }

      auto start_time = std::chrono::high_resolution_clock::now();
      stats_.num_queries += 1;

      lsh_query_.get_transformed_vector(p);

      auto lsh_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_lsh =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              lsh_end_time - start_time);
      stats_.average_lsh_time += elapsed_lsh.count();

      lsh_query_.get_probes_by_table(&tmp_probes_by_table_, num_probes);

      auto multiprobe_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_multiprobe =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              multiprobe_end_time - lsh_end_time);
      stats_.average_multiprobe_time += elapsed_multiprobe.count();

      //for (auto id : tmp_probes_by_table_[0]){
      //  std::cout << id << " ";
      //}
      // std::cout << std::endl;

      hash_table_iterators_ =
          parent_.hash_table_->retrieve_bulk(tmp_probes_by_table_);
      query_counter_ += 1;

      int_fast64_t num_candidates = 0;
      result->clear();
    
      if (max_num_candidates < 0) {
        // set to total number of points.
        max_num_candidates = parent_.n_;
      }
      result->reserve(max_num_candidates);

      while (num_candidates < max_num_candidates &&
             hash_table_iterators_.first != hash_table_iterators_.second) {
        int_fast64_t cur = *(hash_table_iterators_.first);
        if (is_candidate_[cur] != query_counter_) {
          is_candidate_[cur] = query_counter_;
          int_fast32_t table_id = hash_table_iterators_.first.table(),
                       relative_bucket = hash_table_iterators_.first.cur_key_index();  
          ++num_candidates;
          result->push_back({cur, table_id, relative_bucket, /*distance = */ -1});
        }

        ++hash_table_iterators_.first;
      }
      //std::cout << result->size() << std::endl;
      auto hashing_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_hashing =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              hashing_end_time - multiprobe_end_time);
      stats_.average_hash_table_time += elapsed_hashing.count();

      auto sketches_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_sketches =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              sketches_end_time - hashing_end_time);
      stats_.average_sketches_time += elapsed_sketches.count();

      stats_.average_num_candidates += num_candidates;
      stats_.average_num_unique_candidates += result->size();

      auto end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_total =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                    start_time);
      stats_.average_total_query_time += elapsed_total.count();
    }

    void get_duplicate_candidates_and_tables(const QueryType& p,
                                        int_fast64_t num_probes,
                                        int_fast64_t max_num_candidates,
                                        std::vector<FalconnCandidateType>* result) {
      if (result == nullptr) {
        throw LSHTableError("Results vector pointer is nullptr.");
      }

      stats_.num_queries += 1;

      auto start_time = std::chrono::high_resolution_clock::now();

      lsh_query_.get_transformed_vector(p);

      auto lsh_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_lsh =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              lsh_end_time - start_time);
      stats_.average_lsh_time += elapsed_lsh.count();

      lsh_query_.get_probes_by_table(&tmp_probes_by_table_, num_probes);

      auto multiprobe_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_multiprobe =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              multiprobe_end_time - lsh_end_time);
      stats_.average_multiprobe_time += elapsed_multiprobe.count();

      hash_table_iterators_ =
          parent_.hash_table_->retrieve_bulk(tmp_probes_by_table_);

      int_fast64_t num_candidates = 0;
      result->clear();
      if (max_num_candidates < 0) {
        max_num_candidates = std::numeric_limits<int_fast64_t>::max();
      }
      while (num_candidates < max_num_candidates &&
             hash_table_iterators_.first != hash_table_iterators_.second) {
        num_candidates += 1;
        auto cur = *(hash_table_iterators_.first);
        int_fast32_t table_id = hash_table_iterators_.first.table(),
                       relative_bucket = hash_table_iterators_.first.cur_key_index();  
        result->push_back({cur, table_id, relative_bucket, /*distance = */ -1});
        ++hash_table_iterators_.first;
      }

      auto hashing_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_hashing =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              hashing_end_time - multiprobe_end_time);
      stats_.average_hash_table_time += elapsed_hashing.count();

      auto sketches_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_sketches =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              sketches_end_time - hashing_end_time);
      stats_.average_sketches_time += elapsed_sketches.count();

      stats_.average_num_candidates += num_candidates;

      auto end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_total =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                    start_time);
      stats_.average_total_query_time += elapsed_total.count();
    }

    void reset_query_statistics() { stats_.reset(); }

    QueryStatistics get_query_statistics() {
      QueryStatistics res = stats_;
      res.compute_averages();
      return res;
    }

    FalconnMultiprobeType get_transformed_vector(const QueryType& p) {
      lsh_query_.get_transformed_vector(p);
      return lsh_query_.transformed_vector();
    }

    FalconnProbingListType get_probing_sequence(const QueryType& p) {
      return lsh_query_.get_detailed_probing_sequence(p);
    }

    const HashObjectQuery2<LSH, MultiProbe>& lsh_query() const {
      return lsh_query_;
    }
    // TODO: add void get_candidate_sequence(const PointType& p)
    // TODO: add void get_unique_candidate_sequence(const PointType& p)

   private:
    const StaticLSHTable2& parent_;
    int_fast32_t query_counter_ = 0;
    std::vector<int32_t> is_candidate_;
    HashObjectQuery2<LSH, MultiProbe> lsh_query_;
    std::vector<std::vector<HashType>> tmp_probes_by_table_;
    std::pair<typename HashTable::Iterator, typename HashTable::Iterator>
        hash_table_iterators_;

    QueryStatistics stats_;

    void get_unique_candidates_internal(const QueryType& p,
                                        int_fast64_t num_probes,
                                        int_fast64_t max_num_candidates,
                                        std::vector<KeyType>* result) {
      auto start_time = std::chrono::high_resolution_clock::now();

      lsh_query_.get_transformed_vector(p);

      auto lsh_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_lsh =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              lsh_end_time - start_time);
      stats_.average_lsh_time += elapsed_lsh.count();

      lsh_query_.get_probes_by_table(&tmp_probes_by_table_, num_probes);

      auto multiprobe_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_multiprobe =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              multiprobe_end_time - lsh_end_time);
      stats_.average_multiprobe_time += elapsed_multiprobe.count();

      //for (auto id : tmp_probes_by_table_[0]){
      //  std::cout << id << " ";
      //}
      // std::cout << std::endl;

      hash_table_iterators_ =
          parent_.hash_table_->retrieve_bulk(tmp_probes_by_table_);
      query_counter_ += 1;

      int_fast64_t num_candidates = 0;
      result->clear();
      
      if (max_num_candidates < 0) {
        max_num_candidates = std::numeric_limits<int_fast64_t>::max();
      }
      while (num_candidates < max_num_candidates &&
             hash_table_iterators_.first != hash_table_iterators_.second) {
        num_candidates += 1;
        int_fast64_t cur = *(hash_table_iterators_.first);
        if (is_candidate_[cur] != query_counter_) {
          is_candidate_[cur] = query_counter_;
          result->push_back(cur);
        }

        ++hash_table_iterators_.first;
      }
      //std::cout << result->size() << std::endl;
      auto hashing_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_hashing =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              hashing_end_time - multiprobe_end_time);
      stats_.average_hash_table_time += elapsed_hashing.count();

      auto sketches_end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_sketches =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              sketches_end_time - hashing_end_time);
      stats_.average_sketches_time += elapsed_sketches.count();

      stats_.average_num_candidates += num_candidates;
      stats_.average_num_unique_candidates += result->size();
    }
  };

 private:
  int_fast64_t n_;
  const DataStorageType& points_;

  // Used to check the validity of the index file.
  // Only the index file with identical parameters will be loaded.
  struct IndexParameters {
    CompositeHashTableParameters comp_parameters;
    int_fast32_t hash_table_width;
    auto operator<=>(const IndexParameters&) const = default;
  };

  // from and to are id of hash tables
  void setup_table_range(int_fast32_t from, int_fast32_t to,
                         const DataStorageType& points, bool save_index, std::string filename) {
    typename LSH::template BatchHash<int, DataStorageType> bh(*(this->lsh_));
    std::ifstream fin(filename);
    bool index_good = false;
    IndexParameters index_params = {falconn_config.hash_table_params[0], falconn_config.hash_table_width};
    IndexParameters index_params_file;

    if (fin.good()) {
      // load index from storage
      try {
        fin.read(reinterpret_cast<char*>(&index_params_file), sizeof(IndexParameters));
        if (index_params_file != index_params) {
          throw std::runtime_error("Index parameters do not agree. Rebuilding index...");
        }

        for (int_fast32_t ii = from; ii <= to; ++ii) {
          this->hash_table_->add_entries_from_stream(fin, ii);
        }
        index_good = static_cast<bool>(fin);
        std::cout << "Index loaded from " << filename << std::endl;
      } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        std::cout << "Index file corrupted. Rebuilding index..." << std::endl;
      }
    }
    
    if (!index_good) {
      std::ofstream fout;
      if (save_index){
        fout = open_file_create(filename);  // overwrite existing file if corrupted
        fout.write(reinterpret_cast<const char*>(&index_params), sizeof(IndexParameters));
      }
      
      std::vector<HashType> table_hashes;

      for (int_fast32_t ii = from; ii <= to; ++ii) {
        bh.batch_hash_single_table(points, ii, &table_hashes);
        this->hash_table_->reset_table(ii);
        this->hash_table_->add_entries_for_table(table_hashes, ii);
        // does nothing if save_index is false
        
        this->hash_table_->dump_table_to_stream(fout, ii);
      }
    }

  }
};

}  // namespace core
}  // namespace falconn

#endif
