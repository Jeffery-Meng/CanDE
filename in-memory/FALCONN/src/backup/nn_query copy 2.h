#ifndef __NN_QUERY_H__
#define __NN_QUERY_H__

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <iostream>

#include "../oniak/hash_table.hpp"
#include "../oniak/misc.hpp"
#include "../falconn_global.h"
#include "heap.h"
#include "data_storage.h"
#include <limits>

using std::runtime_error;
using std::string;
using std::exception;


namespace falconn {
  /*
 * An auxiliary function that reads a point from a binary file that is produced
 * by a script 'prepare-dataset.sh'
 */
template <typename T>
bool read_point(FILE *file, DenseVector<T> *point) {
  int d;
  if (fread(&d, sizeof(int), 1, file) != 1) {
    return false;
  }
  T *buf = new T[d];
  if (fread(buf, sizeof(T), d, file) != (size_t)d) {
    throw runtime_error("can't read a point");
  }
  point->resize(d);
  for (int i = 0; i < d; ++i) {
    (*point)[i] = buf[i];
  }
  delete[] buf;
  return true;
}

/*
 * An auxiliary function that reads a dataset from a binary file that is
 * produced by a script 'prepare-dataset.sh'
 */
void read_dataset(string file_name, std::vector<DenseVector<float>> *dataset) {
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file) {
    throw runtime_error("can't open the file with the dataset");
  }
  DenseVector<float> p;
  dataset->clear();
  while (read_point(file, &p)) {
    dataset->push_back(p);
  }
  if (fclose(file)) {
    throw runtime_error("fclose() error");
  }
}

namespace core {



template <typename LSHTableQuery, typename LSHTablePointType,
          typename LSHTableKeyType, typename ComparisonPointType,
          typename DistanceType, typename DistanceFunction,
          typename DataStorage>
class NearestNeighborQuery {
  
 public:
 typedef FalconnQueryType QueryType;
  NearestNeighborQuery(LSHTableQuery* table_query,
                       const DataStorage& data_storage)
      : table_query_(table_query), data_storage_(data_storage),
      table_bin_ht_(*falconn_config.seeding_sequence, 0),
      // table_dist_ht_(*falconn_config.seeding_sequence, 0),
      k_(0), num_inserted_(0) {}

  
  KeyType find_nearest_neighbor(const QueryType& q,
                                        const QueryType& q_comp,
                                        int_fast64_t num_probes,
                                        int_fast64_t max_num_candidates) {
    auto start_time = std::chrono::high_resolution_clock::now();

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        &candidates_);
    auto distance_start_time = std::chrono::high_resolution_clock::now();

    // TODO: use nullptr for pointer types
    LSHTableKeyType best_key = -1;

    if (candidates_.size() > 0) {
      typename DataStorage::SubsequenceIterator iter =
          data_storage_.get_subsequence(candidates_);

      best_key = candidates_[0];
      DistanceType best_distance = dst_(q_comp, iter.get_point());
      ++iter;

      // printf("%d %f\n", candidates_[0], best_distance);

      while (iter.is_valid()) {
        DistanceType cur_distance = dst_(q_comp, iter.get_point());
        // printf("%d %f\n", iter.get_key(), cur_distance);
        if (cur_distance < best_distance) {
          best_distance = cur_distance;
          best_key = iter.get_key();
          // printf("  is new best\n");
        }
        ++iter;
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_distance_time += elapsed_distance.count();
    stats_.average_total_query_time += elapsed_total.count();

    return best_key;
  }

  void find_k_nearest_neighbors(const QueryType& q,
                                const QueryType& q_comp,
                                int_fast64_t k, int_fast64_t num_probes,
                                int_fast64_t max_num_candidates,
                                std::vector<LSHTableKeyType>* result) {
    if (result == nullptr) {
      throw NearestNeighborQueryError("Results vector pointer is nullptr.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<LSHTableKeyType>& res = *result;
    res.clear();

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        &candidates_);

    heap_.reset();
    heap_.resize(k);

    auto distance_start_time = std::chrono::high_resolution_clock::now();

    typename DataStorage::SubsequenceIterator iter =
        data_storage_.get_subsequence(candidates_);

    int_fast64_t initially_inserted = 0;
    for (; initially_inserted < k; ++initially_inserted) {
      if (iter.is_valid()) {
        heap_.insert_unsorted(-dst_(q_comp, iter.get_point()), iter.get_key());
        ++iter;
      } else {
        break;
      }
    }

    if (initially_inserted >= k) {
      heap_.heapify();
      while (iter.is_valid()) {
        DistanceType cur_distance = dst_(q_comp, iter.get_point());
        if (cur_distance < -heap_.min_key()) {
          heap_.replace_top(-cur_distance, iter.get_key());
        }
        ++iter;
      }
    }

    res.resize(initially_inserted);
    std::sort(heap_.get_data().begin(),
              heap_.get_data().begin() + initially_inserted);
    for (int_fast64_t ii = 0; ii < initially_inserted; ++ii) {
      res[ii] = heap_.get_data()[initially_inserted - ii - 1].data;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_distance_time += elapsed_distance.count();
    stats_.average_total_query_time += elapsed_total.count();
  }

  void knn_and_kde_infer(const QueryType& q,
                                int_fast64_t k, int_fast64_t num_probes,
                                int_fast64_t max_num_candidates,
                                std::vector<LSHTableKeyType>* result,
                                std::vector<CoordinateType>* kde_result) {
    if (result == nullptr || kde_result == nullptr) {
      throw NearestNeighborQueryError("Results vector pointer is nullptr.");
    }
    auto start_time = std::chrono::high_resolution_clock::now();

    FalconnRange& range = falconn_config.bins_vector[0];
    int num_tables = falconn_config.hash_table_params[0].l;
    int num_bins = range.num_bins();
    if (table_bin_ht_.size() != 131072) table_bin_ht_.resize(131072);
    table_bin_ht_.clear();
    std::vector<int> candidate_num_per_table;

    std::vector<LSHTableKeyType>& res = *result;
    res.clear();

  // Step 1: get duplicate candidates
    table_query_->get_candidates_with_duplicates(q, num_probes, max_num_candidates,
                                        &candidates_, &candidate_num_per_table);
    k_ = k;
    num_inserted_ = 0;
    heap_.reset();
    heap_.resize(k);

    typename DataStorage::SubsequenceIterator iter =
        data_storage_.get_subsequence(candidates_);

    std::vector<double> recalls(num_bins, 0.0);
    std::vector<half_float::half> distances;
    distances.reserve(1'000'000);
    std::vector<int> uniq_cands_per_bin(num_bins, 0);
    // candidates by table
    std::vector<std::vector<int>> tcands_per_bin(num_bins, std::vector<int>(num_tables, 0));  
     // # candidates unique to a given table
    auto utcands_per_bin = tcands_per_bin;    

auto distance_start_time = std::chrono::high_resolution_clock::now();
// Step 2: loop over all candidates, calculate distances of non-duplicate ones, 
// update CanDE counters and min-heaps for nearest neighbors 
    int table = 0, count = 0; // full_count = 0;
    while (iter.is_valid()) {
      
    //for (auto key: candidates_) {
      ONIAK::HashInsertionStatus status = ONIAK::HashInsertionStatus::kNewlyInserted;
      while (count >= candidate_num_per_table[table]) ++table;
      auto key = iter.get_key();
      auto next_key = iter.get_next_key();
      if (next_key > 0) table_bin_ht_.prefetch(next_key);
      auto& table_bin = table_bin_ht_.find_or_insert(key, status);
      // ONIAK::TableBinPair table_bin = {2,2};
      if (status == ONIAK::HashInsertionStatus::kNewlyInserted) {
        // first occurrence of candidate
        DistanceType distance = dst_(q, iter.get_point());
        int bin = range.bin_translate(distance);
        table_bin.bin = bin;
        table_bin.table = table;
        ++tcands_per_bin[bin][table];
        ++utcands_per_bin[bin][table];
        ++uniq_cands_per_bin[bin];
        distances.push_back(static_cast<half_float::half>(distance));
        insert_heap(distance, key);
      } else if (status == ONIAK::HashInsertionStatus::kAlreadyExists) {
        ++tcands_per_bin[table_bin.bin][table];
        if (table_bin.table != 255) {
          --utcands_per_bin[table_bin.bin][table_bin.table];
          table_bin.table = 255;
        }
      } else { //if hash table is full, only check for knn
        //full_count ++;
        if (num_inserted_ < 0) {
          DistanceType distance = dst_(q, iter.get_point());
          insert_heap(distance, key);
       }  // otherwise the result may contain duplicates.
      } 
      ++iter;
      ++count;
    }

     auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    stats_.average_distance_time += elapsed_distance.count();

    // step 3: return top Knn results
    int heap_size = (num_inserted_ < 0)? k: num_inserted_;
    res.resize(heap_size);
    std::sort(heap_.get_data().begin(),
              heap_.get_data().begin() + heap_size);
    for (int_fast64_t ii = 0; ii < heap_size; ++ii) {
      res[ii] = heap_.get_data()[heap_size - ii - 1].data;
    }

   

  // step 4: infer recalls of each bin
    for (int bin = 0; bin < num_bins; ++bin) {
      std::vector<double> rhos(num_tables, 0.0);
      for (int table = 0; table < num_tables; ++table) {
        rhos[table] = static_cast<double>(tcands_per_bin[bin][table] - utcands_per_bin[bin][table]) /
            static_cast<double>(uniq_cands_per_bin[bin] - utcands_per_bin[bin][table]);
        
      }
      double rho = ONIAK::final_prob(rhos);
      recalls[bin] = rho;
    }

  // step 5: infer KDE
    int gamma_num = falconn_config.gamma.size();
    kde_result->assign(gamma_num, 0.0);
    for (auto& distance: distances) {
      int bin = range.bin_translate(distance);
      for (int gammaid = 0; gammaid < gamma_num; ++gammaid) {
        if (recalls[bin] > 1e-4) { // avoid division by zero
          kde_result->operator[](gammaid) += ONIAK::kde(distance, falconn_config.gamma[gammaid]) / recalls[bin];
        }
      }
    }

    auto sketches_end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_sketches =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              sketches_end_time - end_time);
    stats_.average_sketches_time += elapsed_sketches.count();
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(sketches_end_time -
                                                                  start_time);
    stats_.average_total_query_time += elapsed_total.count();
    //std::cout << full_count << std::endl;
  }

  void find_near_neighbors(const QueryType& q,
                           const QueryType& q_comp,
                           DistanceType threshold, int_fast64_t num_probes,
                           int_fast64_t max_num_candidates,
                           std::vector<LSHTableKeyType>* result) {
    if (result == nullptr) {
      throw NearestNeighborQueryError("Results vector pointer is nullptr.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<LSHTableKeyType>& res = *result;
    res.clear();

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        &candidates_);
    auto distance_start_time = std::chrono::high_resolution_clock::now();

    typename DataStorage::SubsequenceIterator iter =
        data_storage_.get_subsequence(candidates_);
    while (iter.is_valid()) {
      DistanceType cur_distance = dst_(q_comp, iter.get_point());
      if (cur_distance < threshold) {
        res.push_back(iter.get_key());
      }
      ++iter;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_distance_time += elapsed_distance.count();
    stats_.average_total_query_time += elapsed_total.count();
  }

  void get_candidates_with_duplicates(const QueryType& q,
                                      int_fast64_t num_probes,
                                      int_fast64_t max_num_candidates,
                                      std::vector<FalconnCandidateType>* result) {
    auto start_time = std::chrono::high_resolution_clock::now();

    table_query_->get_duplicate_candidates_and_tables(q, num_probes,
                                                 max_num_candidates, result);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_total_query_time += elapsed_total.count();
  }

  void get_unique_candidates(const QueryType& q,
                             int_fast64_t num_probes,
                             int_fast64_t max_num_candidates,
                             std::vector<FalconnCandidateType>* result) {
    auto start_time = std::chrono::high_resolution_clock::now();

    table_query_->get_unique_candidates_and_tables(q, num_probes, max_num_candidates,
                                        result);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_total =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time);
    stats_.average_total_query_time += elapsed_total.count();
  }

  void reset_query_statistics() {
    table_query_->reset_query_statistics();
    stats_.reset();
  }

  QueryStatistics get_query_statistics() {
    QueryStatistics res = table_query_->get_query_statistics();
    res.average_total_query_time = stats_.average_total_query_time;
    res.average_distance_time = stats_.average_distance_time;
    res.average_sketches_time = stats_.average_sketches_time;

    if (res.num_queries > 0) {
      res.average_total_query_time /= res.num_queries;
      res.average_distance_time /= res.num_queries;
      res.average_sketches_time /= res.num_queries;
    }
    return res;
  }

  FalconnMultiprobeType get_transformed_vector(const QueryType& q) {
    return table_query_->get_transformed_vector(q);
  }

  FalconnProbingListType get_probing_sequence(const QueryType& q) {
    return table_query_->get_probing_sequence(q);
  }

 private:
  LSHTableQuery* table_query_;
  const DataStorage& data_storage_;
  std::vector<LSHTableKeyType> candidates_;
  DistanceFunction dst_;
  SimpleHeap<DistanceType, LSHTableKeyType> heap_;
  ONIAK::ONIAKHT<int, ONIAK::TableBinPair, uint16_t, 80> table_bin_ht_;
  // ONIAK::ONIAKHT<int, ONIAK::TableDistancePair, uint16_t, 96> table_dist_ht_;
  QueryStatistics stats_;
  // number of nearest neighbors to return
  int k_, num_inserted_;

  void insert_heap(DistanceType distance, LSHTableKeyType key) {
    if (num_inserted_ >= 0) {
      heap_.insert_unsorted(-distance, key);
      ++num_inserted_;
      if (num_inserted_ == k_) {
        heap_.heapify();
        num_inserted_ = -1;
      }
    } else if (distance < -heap_.min_key()) {
      heap_.replace_top(-distance, key);
    }
    
  }
};

}  // namespace core
}  // namespace falconn

#endif
