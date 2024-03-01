#ifndef __NN_QUERY_CACHE_H__
#define __NN_QUERY_CACHE_H__

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

#include "../falconn_global.h"
#include "../homogenizer.h"
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

namespace core {

template <typename LSHTableQuery, typename LSHTablePointType,
          typename LSHTableKeyType, typename ComparisonPointType,
          typename DistanceType, typename DistanceFunction,
          typename DataStorage>
class CachedNearestNeighborQuery {
  
 public:
 struct CachedMatrixEntry {
   int matid = 0;
   PointType Ax;
 };

 typedef PointType QueryType;

  void update_A(MatrixType&& a) {
    ++matrix_id_;
    // This invalidates all caches.
    matrix_ = a;
  }

  CachedNearestNeighborQuery(LSHTableQuery* table_query,
                       const DataStorage& data_storage)
      : table_query_(table_query), data_storage_(data_storage), matrix_id_(0), caches_(data_storage.size()) {}

  LSHTableKeyType find_nearest_neighbor(const QueryType& q,
                                        const QueryType& q_comp,
                                        int_fast64_t num_probes,
                                        int_fast64_t max_num_candidates) {
    return -1; 
    // This function is not supported
    //return best_key;
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

    table_query_->get_unique_candidates(query_from_point(matrix_, q), num_probes, max_num_candidates,
                                        &candidates_);

    heap_.reset();
    heap_.resize(k);

    auto distance_start_time = std::chrono::high_resolution_clock::now();

    typename DataStorage::SubsequenceIterator iter =
        data_storage_.get_subsequence(candidates_);

    int_fast64_t initially_inserted = 0;
    for (; initially_inserted < k; ++initially_inserted) {
      if (iter.is_valid()) {
        heap_.insert_unsorted(-compute_distance(iter, q_comp), iter.get_key());
        ++iter;
      } else {
        break;
      }
    }

    if (initially_inserted >= k) {
      heap_.heapify();
      while (iter.is_valid()) {
        DistanceType cur_distance = compute_distance(iter, q_comp);
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

  void find_near_neighbors(const QueryType& q,
                           const QueryType& q_comp,
                           DistanceType threshold, int_fast64_t num_probes,
                           int_fast64_t max_num_candidates,
                           std::vector<LSHTableKeyType>* result) {
    // This function is not supported
  }

  void get_candidates_with_duplicates(const QueryType& q,
                                      int_fast64_t num_probes,
                                      int_fast64_t max_num_candidates,
                                      std::vector<LSHTableKeyType>* result) {
    // This function is not supported
  }

  void reset_query_statistics() {
    table_query_->reset_query_statistics();
    stats_.reset();
  }

  QueryStatistics get_query_statistics() {
    QueryStatistics res = table_query_->get_query_statistics();
    res.average_total_query_time = stats_.average_total_query_time;
    res.average_distance_time = stats_.average_distance_time;

    if (res.num_queries > 0) {
      res.average_total_query_time /= res.num_queries;
      res.average_distance_time /= res.num_queries;
    }
    return res;
  }

 private:
  LSHTableQuery* table_query_;
  const DataStorage& data_storage_;
  std::vector<LSHTableKeyType> candidates_;
  DistanceFunction dst_;
  SimpleHeap<DistanceType, LSHTableKeyType> heap_;
  int matrix_id_;
  std::vector<CachedMatrixEntry> caches_;
  MatrixType matrix_;

  QueryStatistics stats_;

  bool is_valid(const CachedMatrixEntry& entry) const{
    return matrix_id_ == entry.matid;
  }

  float compute_distance(const typename DataStorage::SubsequenceIterator& iter, const QueryType& q) {
    auto key = iter.get_key();
    if (caches_[key].matid != matrix_id_) {
      caches_[key].matid = matrix_id_;
      caches_[key].Ax = matrix_ * iter.get_point();
    }
    return dst_(q, caches_[key].Ax);
  }
};



}  // namespace core
}  // namespace falconn

#endif
