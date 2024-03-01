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
#include "../oniak/real_array.hpp"
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
  enum class Task {kKDE=1, kQDDE=2};

  NearestNeighborQuery(LSHTableQuery* table_query,
                       const DataStorage& data_storage)
      : table_query_(table_query), data_storage_(data_storage),
      mp_recalls_(falconn_config.mp_prob_filename),
      table_bin_ht_(*falconn_config.seeding_sequence, falconn_config.cande_table_size),
      k_(0), num_inserted_(0) {
        // raise single-table Multiprobe recalls to that of L tables
        int num_tables = falconn_config.hash_table_params[0].l;
        float bucket_width = falconn_config.hash_table_params[0].bucket_width;
        for (auto& val: mp_recalls_.array()) {
          val = 1.0 - std::pow(1.0 - val, num_tables);
        }
        // normalize distance by bucket width
        mp_recalls_.start() *= bucket_width;
        mp_recalls_.step() *= bucket_width;
      }

  
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
    std::vector<float> distances;
    distances.reserve(candidates_.size());
    std::vector<int> uniq_cands_per_bin(num_bins, 0);
    // candidates by table
    std::vector<std::vector<int>> tcands_per_bin(num_bins, std::vector<int>(num_tables, 0));  
    int gamma_num = falconn_config.gamma.size();
    std::vector<float> gammas(gamma_num);
    std::transform(falconn_config.gamma.begin(), falconn_config.gamma.end(), gammas.begin(),
      [](float x) {return x * x * 2.0;});
    std::vector<std::vector<float>> kde_sums(gamma_num, std::vector<float>(num_bins, 0.0));
     // # candidates unique to a given table
    auto utcands_per_bin = tcands_per_bin;    

auto distance_start_time = std::chrono::high_resolution_clock::now();
// Step 2a: compute all distances
    int table = 0, count = 0; // full_count = 0;
    while (iter.is_valid()) {
      DistanceType distance = dst_(q, iter.get_point());
      distances.push_back(distance);
      ++iter;
    }

// Step 2b: deduplicate using hash tables, and update inference counters
    for (auto key: candidates_) {
      ONIAK::HashInsertionStatus status = ONIAK::HashInsertionStatus::kNewlyInserted;
      float distance = distances[count];
      int bin = range.bin_translate(distance);
      while (count >= candidate_num_per_table[table]) ++table;
      auto& table_find = table_bin_ht_.find_or_insert(key, status);
      // ONIAK::TableBinPair table_bin = {2,2};
      if (status == ONIAK::HashInsertionStatus::kNewlyInserted) {
        // first occurrence of candidate
        // table_bin.bin = bin;
        table_find = table;
        ++tcands_per_bin[bin][table];
        ++utcands_per_bin[bin][table];
        ++uniq_cands_per_bin[bin];
        for (int gammaid = 0; gammaid < gamma_num; ++ gammaid) {
          kde_sums[gammaid][bin] += ONIAK::kdev2(distance, gammas[gammaid]);
        }
        insert_heap(distance, key);
      } else if (status == ONIAK::HashInsertionStatus::kAlreadyExists) {
        ++tcands_per_bin[bin][table];
        if (table_find != 255) {
          --utcands_per_bin[bin][table_find];
          table_find = 255;
        }
      } else { //if hash table is full, only check for knn
        if (num_inserted_ < 0) {
          insert_heap(distance, key);
       }  // otherwise the result may contain duplicates.
      } 
      ++count;
    }

     auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    stats_.average_distance_time += elapsed_distance.count();

    // step 3: return top Knn results
    dump_knn(res);

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
    kde_result->assign(gamma_num, 0.0);
    for (int gammaid = 0; gammaid < gamma_num; ++gammaid) {
      for (int bin = 0; bin < num_bins; ++bin) {
        if (recalls[bin] > 1e-4) {
          kde_result->operator[](gammaid) += kde_sums[gammaid][bin] / recalls[bin];
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

  // CanDE infer implementation using associative memory, 
  // this is fast only for small datasets
   void cande_infer_associative(const QueryType& q,
                                int_fast64_t k, int_fast64_t num_probes,
                                int_fast64_t max_num_candidates,
                                std::vector<LSHTableKeyType>* result,
                                std::vector<CoordinateType>* kde_result,
                                Task task) {
    if (result == nullptr || kde_result == nullptr) {
      throw NearestNeighborQueryError("Results vector pointer is nullptr.");
    }
    auto start_time = std::chrono::high_resolution_clock::now();

    FalconnRange& range = falconn_config.bins_vector[0];
    int num_bins = range.num_bins();
    int num_values = (task == Task::kKDE)? falconn_config.gamma.size() : num_bins;
    int num_sums = (task == Task::kKDE)? falconn_config.gamma.size() : 1;
    std::vector<int> candidate_num_per_table;
    int num_tables = falconn_config.hash_table_params[0].l;

    std::vector<LSHTableKeyType>& res = *result;
    res.clear();
    kde_result->assign(num_values, 0);

  // Step 1: get duplicate candidates
    table_query_->get_candidates_with_duplicates(q, num_probes, max_num_candidates,
                                        &candidates_, &candidate_num_per_table);
    k_ = k;
    num_inserted_ = 0;
    heap_.reset();
    heap_.resize(k);    

    std::vector<double> recalls(num_bins, 0.0);
    std::vector<LSHTableKeyType> dedup_candidates;
    dedup_candidates.reserve(candidates_.size());
    std::vector<float> distances(falconn_config.num_points);
    // stores the table in which this candidate first appears
    std::vector<uint8_t> cand_tables(falconn_config.num_points, 255);
    std::vector<int> uniq_cands_per_bin(num_bins, 0);
    // candidates by table
    std::vector<std::vector<int>> tcands_per_bin(num_bins, std::vector<int>(num_tables, 0));  
     // # candidates unique to a given table
    auto utcands_per_bin = tcands_per_bin;    
    std::vector<std::vector<float>> kde_sums(num_sums, std::vector<float>(num_bins, 0.0));
    std::vector<float> gammas(num_values);
    if (task == Task::kKDE) {
      std::transform(falconn_config.gamma.begin(), falconn_config.gamma.end(), gammas.begin(),
        [](float x) {return x * x * 2.0;});
    }

auto distance_start_time = std::chrono::high_resolution_clock::now();
// Step 2z: deduplicate
    int table = 0, count = 0;
    for (auto cand : candidates_) {
      while (count >= candidate_num_per_table[table]) ++table;
      if (cand_tables[cand] == 255) { // new item {
        cand_tables[cand] = table;
        dedup_candidates.push_back(cand);
      }
      ++count;
    }

typename DataStorage::SubsequenceIterator iter =
        data_storage_.get_subsequence(dedup_candidates);
// Step 2a: compute all distances
     // full_count = 0;
    while (iter.is_valid()) {
      DistanceType distance = dst_(q, iter.get_point());
      distances[iter.get_key()] = distance;
      insert_heap(distance, iter.get_key());
      ++iter;
    }

// Step 2b: deduplicate using hash tables, and update inference counters
    table = 0;
    count = 0;
    for (auto key: candidates_) {
      while (count >= candidate_num_per_table[table]) ++table;
      int bin = range.bin_translate(distances[key]);
      if (cand_tables[key] == table) {
        // first occurrence of candidate
        ++tcands_per_bin[bin][table];
        ++utcands_per_bin[bin][table];
        ++uniq_cands_per_bin[bin];
        if (task == Task::kKDE) {
          for (int gammaid = 0; gammaid < num_values; ++gammaid) {
            kde_sums[gammaid][bin] += ONIAK::kdev2(distances[key], gammas[gammaid]) / recalls[bin];
          }
        } else {
          kde_sums[0][bin] += 1.0;
        }
      } else {  // duplicate
        ++tcands_per_bin[bin][table];
        if (cand_tables[key] != 254) {
          --utcands_per_bin[bin][cand_tables[key]];
        }
        cand_tables[key] = 254;
      } 
      ++count;
    }

     auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    stats_.average_distance_time += elapsed_distance.count();

    // step 3: return top Knn results
    dump_knn(res);

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

  // step 5: infer KDE or QDDE

    for (int gammaid = 0; gammaid < num_sums; ++gammaid) {
      for (int bin = 0; bin < num_bins; ++bin) {
        if (task == Task::kKDE) {
          if (recalls[bin] > 1e-4) { // avoid division by zero
            kde_result->operator[](gammaid) += kde_sums[gammaid][bin] / recalls[bin];
          }
        } else {  // QDDE
          if (recalls[bin] > 1e-4) { // avoid division by zero
            kde_result->operator[](bin) += kde_sums[gammaid][bin] / recalls[bin];
          }
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

  void knn_and_qdde_infer(const QueryType& q,
                                int_fast64_t k, int_fast64_t qid,int_fast64_t num_probes,
                                int_fast64_t max_num_candidates,
                                std::vector<LSHTableKeyType>* result,
                                std::vector<CoordinateType>* histogram) {
    if (result == nullptr || histogram == nullptr) {
      throw NearestNeighborQueryError("Results vector pointer is nullptr.");
    }
    auto start_time = std::chrono::high_resolution_clock::now();

    FalconnRange& range = falconn_config.bins_vector[0];
    int num_tables = falconn_config.hash_table_params[0].l;
    int num_bins = range.num_bins();
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
    histogram->assign(num_bins, 0.0);

    typename DataStorage::SubsequenceIterator iter =
        data_storage_.get_subsequence(candidates_);
    std::vector<float> distances;
    distances.reserve(candidates_.size());
    std::vector<int> uniq_cands_per_bin(num_bins, 0);
    // candidates by table
    std::vector<std::vector<int>> tcands_per_bin(num_bins, std::vector<int>(num_tables, 0));  
     // # candidates unique to a given table
    auto utcands_per_bin = tcands_per_bin;    

auto distance_start_time = std::chrono::high_resolution_clock::now();
// Step 2a: compute all distances
    int table = 0, count = 0; // full_count = 0;
    while (iter.is_valid()) {
      DistanceType distance = dst_(q, iter.get_point());
      distances.push_back(distance);
      ++iter;
    }

// Step 2b: deduplicate using hash tables, and update inference counters
    for (auto key: candidates_) {
      ONIAK::HashInsertionStatus status = ONIAK::HashInsertionStatus::kNewlyInserted;
      float distance = distances[count];
      int bin = range.bin_translate(distance);
      while (count >= candidate_num_per_table[table]) ++table;
      auto& table_find = table_bin_ht_.find_or_insert(key, status);
      // ONIAK::TableBinPair table_bin = {2,2};
      if (status == ONIAK::HashInsertionStatus::kNewlyInserted) {
        // first occurrence of candidate
        // Bin
        // table_bin.bin = bin;
        table_find = table;
        ++tcands_per_bin[bin][table];
        ++utcands_per_bin[bin][table];
        ++uniq_cands_per_bin[bin];
        insert_heap(distance, key);
      } else if (status == ONIAK::HashInsertionStatus::kAlreadyExists) {
        ++tcands_per_bin[bin][table];
        if (table_find != 255) {
          --utcands_per_bin[bin][table_find];
          table_find = 255;
        }
      } else { //if hash table is full, only check for knn
        if (num_inserted_ < 0) {
          insert_heap(distance, key);
       }  // otherwise the result may contain duplicates.
      } 
      ++count;
    }

     auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_distance =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - distance_start_time);
    stats_.average_distance_time += elapsed_distance.count();

    // step 3: return top Knn results
    dump_knn(res);

  // step 4: infer histogram of each bin
    for (int bin = 0; bin < num_bins; ++bin) {
      std::vector<double> rhos(num_tables, 0.0);
      for (int table = 0; table < num_tables; ++table) {
        rhos[table] = static_cast<double>(tcands_per_bin[bin][table] - utcands_per_bin[bin][table]) /
            static_cast<double>(uniq_cands_per_bin[bin] - utcands_per_bin[bin][table]);
        
      }
      double rho = ONIAK::final_prob(rhos);
      histogram->operator[](bin) = uniq_cands_per_bin[bin] / rho;
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

  void cande_precomputed(const QueryType& q,
                                int_fast64_t k, int_fast64_t num_probes,
                                int_fast64_t max_num_candidates,
                                std::vector<LSHTableKeyType>* result,
                                std::vector<CoordinateType>* val_result,
                                Task task) {
    if (result == nullptr || val_result == nullptr) {
      throw NearestNeighborQueryError("Results vector pointer is nullptr.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<LSHTableKeyType>& res = *result;
    res.clear();
    FalconnRange& range = falconn_config.bins_vector[0];
    int num_bins = range.num_bins();
    int num_values = (task == Task::kKDE)? falconn_config.gamma.size() : num_bins;

    table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                        &candidates_);

    k_=k;
    heap_.reset();
    heap_.resize(k);
    val_result->assign(num_values, 0);

    auto distance_start_time = std::chrono::high_resolution_clock::now();

    typename DataStorage::SubsequenceIterator iter =
        data_storage_.get_subsequence(candidates_);
    std::vector<float> distances;
    distances.reserve(candidates_.size());
    while (iter.is_valid()) {
      DistanceType cur_distance = dst_(q, iter.get_point());
      insert_heap(cur_distance, iter.get_key());
      distances.push_back(cur_distance);
      ++iter;
    }
    
    std::vector<float> gammas(num_values);
    if (task == Task::kKDE) {
      std::transform(falconn_config.gamma.begin(), falconn_config.gamma.end(), gammas.begin(),
        [](float x) {return x * x * 2.0;});
    }

    for (float cur_distance: distances) {
      if (task == Task::kKDE) {
        for (size_t gammaid = 0; gammaid < val_result->size(); ++gammaid) {
          val_result->operator[](gammaid) += 
              ONIAK::kdev2(cur_distance, gammas[gammaid]) / mp_recalls_[cur_distance];
        }
      } else {  // task is QDDE
        int bin = range.bin_translate(cur_distance);
        val_result->operator[](bin) += 1.0 / mp_recalls_[cur_distance];
      } 
    }
    
    dump_knn(res);

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

  // Function to output candidates per table
  std::vector<int>  knn_candidate(const QueryType& q,
                                  int_fast64_t k, int_fast64_t num_probes,
                                  int_fast64_t max_num_candidates) {
      std::vector<int> candidate_num_per_table;


    // Step 1: get duplicate candidates
      table_query_->get_candidates_with_duplicates(q, num_probes, max_num_candidates,
                                          &candidates_, &candidate_num_per_table);
    
      return candidate_num_per_table;
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
  // Precompute multiproed recalls
  ONIAK::RealArray mp_recalls_;

  ONIAK::ONIAKHT<int, uint8_t, uint16_t, 64> table_bin_ht_;
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

  void dump_knn(std::vector<LSHTableKeyType>& res) {
    int heap_size = (num_inserted_ < 0)? k_: num_inserted_;
    res.resize(heap_size);
    std::sort(heap_.get_data().begin(),
              heap_.get_data().begin() + heap_size);
    for (int_fast64_t ii = 0; ii < heap_size; ++ii) {
      res[ii] = heap_.get_data()[heap_size - ii - 1].data;
    }
  }
};

}  // namespace core
}  // namespace falconn

#endif
