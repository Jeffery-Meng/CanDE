#ifndef __CPP_WRAPPER_IMPL_H__
#define __CPP_WRAPPER_IMPL_H__

#include <atomic>
#include <random>
#include <thread>
#include <type_traits>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <memory>
#include <cstring>
#include <Exception.h>

#include "data_storage_adapter.h"
#include "../falconn_global.h"
#include "../lsh_nn_table.h"

#include "../core/bit_packed_flat_hash_table.h"
#include "../core/composite_hash_table.h"
#include "../core/cosine_distance.h"
#include "../core/data_storage.h"
#include "../core/euclidean_distance.h"
#include "../core/flat_hash_table.h"
#include "../core/gaussian_hash.h"
#include "../core/multiprobe.h"
#include "../core/hyperplane_hash.h"
#include "../core/lsh_table.h"
#include "../core/lsh_table_new.h"
#include "../core/nn_query.h"
#include "../core/polytope_hash.h"
#include "../core/probing_hash_table.h"
#include "../core/stl_hash_table.h"
#include "../core/l1_distance.h"
#include "../core/rangesummable.h"
#include "../core/partition_metric.h"
#include "../core/partition_hash_table.h"
#include "../core/nearest_hyperplane.h"
#include "../core/math_helpers.h"
#include "../core/heap.h"

namespace falconn {
namespace wrapper {

template <typename T>
inline constexpr bool false_constexpr = false;

template <typename PointType, typename KeyType, typename DistanceType,
  typename LSHTable, typename ScalarType, typename DistanceFunction,
  typename DataStorage>
class LSHNNQueryWrapper : public LSHNearestNeighborQuery<PointType, KeyType> {
public:
  using NNQueryType = FalconnNNQueryType;

  LSHNNQueryWrapper(const LSHTable& parent, int_fast64_t num_probes,
                    int_fast64_t max_num_candidates,
                    const DataStorage& data_storage)
    : num_probes_(num_probes), max_num_candidates_(max_num_candidates) {
    if (num_probes <= 0) {
      throw LSHNearestNeighborTableError(
        "Number of probes must be at least 1.");
    }
    internal_query_.reset(new typename LSHTable::Query(parent, num_probes));
    internal_nn_query_.reset(
      new NNQueryType(internal_query_.get(), data_storage));
  }

  KeyType find_nearest_neighbor(const FalconnQueryType& q) override {
    return internal_nn_query_->find_nearest_neighbor(q, q, num_probes_,
                                                     max_num_candidates_);
  }

  void find_k_nearest_neighbors(const FalconnQueryType& q, int_fast64_t k,
                                std::vector<KeyType>* result) override {
    internal_nn_query_->find_k_nearest_neighbors(q, q, k, num_probes_,
                                                 max_num_candidates_, result);
  }

  FalconnMultiprobeType get_transformed_vector(const FalconnQueryType& q) {
    return internal_nn_query_->get_transformed_vector(q);
  }

  FalconnProbingListType get_probing_sequence(const FalconnQueryType& q) {
    return internal_nn_query_->get_probing_sequence(q);
  }

  void find_near_neighbors(const FalconnQueryType& q, DistanceType threshold,
                           std::vector<KeyType>* result) override {
    internal_nn_query_->find_near_neighbors(q, q, threshold, num_probes_,
                                            max_num_candidates_, result);
  }

  void get_candidates_with_duplicates(const FalconnQueryType& q,
                                      std::vector<FalconnCandidateType>* result) override {
    internal_nn_query_->get_candidates_with_duplicates(
      q, num_probes_, max_num_candidates_, result);
  }

  void get_unique_candidates(const FalconnQueryType& q, std::vector<FalconnCandidateType>* result) override {
    internal_nn_query_->get_unique_candidates(q, num_probes_,
                                              max_num_candidates_, result);
  }

  void knn_and_kde_infer(const FalconnQueryType& q, int qid, int_fast64_t k, std::vector<KeyType>* knn_result,
                         std::vector<CoordinateType>* kde_result) {
    internal_nn_query_->knn_and_kde_infer(q, qid, k, num_probes_,
                                          max_num_candidates_, knn_result, kde_result);
  }

  void knn_and_precomputed(const FalconnQueryType& q, int qid, int_fast64_t k, std::vector<KeyType>* knn_result,
                           std::vector<CoordinateType>* kde_result, CanDETask task) {
    internal_nn_query_->cande_precomputed(q, qid, k, num_probes_,
                                          max_num_candidates_, knn_result, kde_result, task);
  }
  void knn_and_cande_cp_adjusted(const FalconnQueryType& q, int qid, int_fast64_t k,
                                 std::vector<KeyType>* knn_result,
                                 std::vector<CoordinateType>* kde_result, CanDETask task) {
    internal_nn_query_->knn_and_cande_cp_adjusted(q, qid, k, num_probes_,
                                                  max_num_candidates_, knn_result, kde_result, task);
  }
  void knn_and_cande_resample(const FalconnQueryType& q, int qid, int_fast64_t k,
                              std::vector<KeyType>* knn_result,
                              std::vector<CoordinateType>* kde_result, CanDETask task) {
    internal_nn_query_->knn_and_cande_resample(q, qid, k, num_probes_,
                                               max_num_candidates_, knn_result, kde_result, task);
  }

  std::vector<int> get_knn_candidate(const FalconnQueryType& q, int_fast64_t k) {
    return internal_nn_query_->knn_candidate(q, k, num_probes_,
                                             max_num_candidates_);
  }

  void knn_infer_associative(const FalconnQueryType& q, int qid, int_fast64_t k, std::vector<KeyType>* knn_result,
                             std::vector<CoordinateType>* kde_result, CanDETask task) {
    internal_nn_query_->cande_infer_associative(q, qid, k, num_probes_,
                                                max_num_candidates_, knn_result, kde_result, task);
  }

  void knn_and_qdde_infer(const FalconnQueryType& q, int qid, int_fast64_t k, std::vector<KeyType>* knn_result,
                          std::vector<CoordinateType>* histogram, CanDETask task) {
    internal_nn_query_->knn_and_qdde_infer(q, qid, k, num_probes_,
                                           max_num_candidates_, knn_result, histogram, task);
  }

  int_fast64_t get_num_probes() override { return num_probes_; }

  void set_num_probes(int_fast64_t new_num_probes) override {
    if (new_num_probes <= 0) {
      throw LSHNearestNeighborTableError(
        "Number of probes must be at least 1.");
    }
    num_probes_ = new_num_probes;
  }

  int_fast64_t get_max_num_candidates() override { return max_num_candidates_; }

  void set_max_num_candidates(int_fast64_t new_max_num_candidates) override {
    max_num_candidates_ = new_max_num_candidates;
  }

  void reset_query_statistics() override {
    internal_nn_query_->reset_query_statistics();
  }

  QueryStatistics get_query_statistics() override {
    return internal_nn_query_->get_query_statistics();
  }

  // without_threshold
  void find_k_nearest_neighbors_without_threshold(const FalconnQueryType& q, int_fast64_t k,
                                                  std::vector<KeyType>* result, unsigned q_cnt) override {
    find_k_nearest_neighbors(q, k, result);
  }

  // with_threshold
  void find_k_nearest_neighbors_with_threshold(const FalconnQueryType& q, int_fast64_t k,
                                               std::vector<KeyType>* result, unsigned q_cnt, float t) override {
    find_k_nearest_neighbors(q, k, result);
  }

  virtual ~LSHNNQueryWrapper() {}

  const typename LSHTable::Query* internal_query() const {
    return internal_query_.get();
  }

protected:
  std::unique_ptr<typename LSHTable::Query> internal_query_;
  std::unique_ptr<NNQueryType> internal_nn_query_;
  int_fast64_t num_probes_;
  int_fast64_t max_num_candidates_;
};

template <typename PointType, typename KeyType, typename DistanceType,
  typename DistanceFunction, typename LSHTable, typename LSHFunction,
  typename HashTableFactory, typename CompositeHashTable,
  typename DataStorage>
class LSHNNTableWrapper : public LSHNearestNeighborTable<PointType, KeyType> {
public:
  LSHNNTableWrapper(std::vector<std::unique_ptr<LSHFunction>> lshes,
                    std::unique_ptr<LSHTable> lsh_table,
                    std::unique_ptr<HashTableFactory> hash_table_factory,
                    std::vector<std::unique_ptr<CompositeHashTable>> composite_tables,
                    std::unique_ptr<DataStorage> data_storage)
    : lshes_(std::move(lshes)),
    lsh_table_(std::move(lsh_table)),
    hash_table_factory_(std::move(hash_table_factory)),
    composite_tables_(std::move(composite_tables)),
    data_storage_(std::move(data_storage)) {}

  void add_table() override {} // deleted for now
  /*  lsh_->add_table();
    composite_hash_table_vec_->add_table();
    lsh_table_->add_table();
  }*/

  // Here, num_probes is the maximum number of probes, used in the precomputation of MultiProbe
  std::unique_ptr<FalconnQueryWrapper>
    construct_query_object(int_fast64_t num_probes,
                           int_fast64_t max_num_candidates, unsigned num_filters = 0, float recall_target = 0.)
    const override {
    assert(num_probes > 0);

    typedef CoordinateType ScalarType;
    if constexpr (std::is_same_v<FalconnQueryWrapper, LSHNNQueryWrapper<
                  PointType, KeyType, DistanceType, LSHTable,
                  ScalarType, DistanceFunction, DataStorage>>) {
      std::unique_ptr<FalconnQueryWrapper>
        nn_query(new FalconnQueryWrapper(
          *lsh_table_, num_probes, max_num_candidates, *data_storage_));
      return nn_query;
    } else {
      static_assert(false_constexpr<PointType>, "Unsupported Query Wrapper type!");
    }
  }

  ~LSHNNTableWrapper() {}

protected:
  std::vector<std::unique_ptr<LSHFunction>> lshes_;
  std::unique_ptr<LSHTable> lsh_table_;
  std::unique_ptr<HashTableFactory> hash_table_factory_;
  std::vector<std::unique_ptr<CompositeHashTable>> composite_tables_;
  std::unique_ptr<DataStorage> data_storage_;
};

template <typename PointType, typename KeyType, typename PointSet>
class StaticTableFactory {
public:
  typedef CoordinateType ScalarType;

  typedef typename DataStorageAdapter<PointSet>::template DataStorage<KeyType>
    DataStorageType;

  // using CompositeTableT  = CompositeTable<HashType, KeyType, HashTable>;

  StaticTableFactory(const PointSet& points,
                     const LSHConstructionParameters& params)
    : points_(points), params_(params) {}

  std::unique_ptr<LSHNearestNeighborTable<PointType, KeyType>> setup() {
    if (params_.dimension < 1) {
      throw LSHNNTableSetupError(
        "Point dimension must be at least 1. Maybe "
        "you forgot to set the point dimension in the parameter struct?");
    }

    if (params_.num_setup_threads < 0) {
      throw LSHNNTableSetupError(
        "The number of setup threads cannot be "
        "negative. Maybe you forgot to set num_setup_threads in the "
        "parameter struct? A value of 0 indicates that FALCONN should use "
        "the maximum number of available hardware threads.");
    }

    data_storage_ = std::move(
      DataStorageAdapter<PointSet>::template construct_data_storage<KeyType>(
        points_));

    // ComputeNumberOfHashBits<PointType> helper;
    // num_bits_ = 22; // Current hash bits is set to 20;
    num_bits_ = params_.hash_table_width;
    n_ = data_storage_->size();
    assert(n_ == params_.num_points && n_ > 0);
    assert(data_storage_->dimension() == params_.dimension);

    // std::unique_ptr<LSH> lsh;
    std::unique_ptr<typename HashTable::Factory> factory(
      new typename HashTable::Factory(1 << num_bits_));
    std::vector<std::unique_ptr<CompositeTable>> partition_tables;
    std::vector<std::unique_ptr<LSH>> partition_lshes;
    std::unique_ptr<LSHTableType> lsh_table;

    assert(params_.num_partitions == static_cast<int_fast32_t>(params_.hash_table_params.size()));

    for (const auto& par_para : params_.hash_table_params) {
      if constexpr (std::is_same_v<CompositeTable, core::StaticPartition<HashType, KeyType,
                    PartitionMetric::PartitionMetricType, HashTable>>) {
        partition_tables.emplace_back(std::make_unique<CompositeTable>(
          par_para.l, factory.get(), par_para.partition_lower, par_para.partition_upper));
      } else if constexpr (std::is_same_v<CompositeTable, core::StaticCompositeHashTable2<HashType, KeyType, HashTable>>) {
        assert(params_.hash_table_params.size() == 1 && "Partitioning is disabled.");
        partition_tables.emplace_back(std::make_unique<CompositeTable>(par_para.l, factory.get()));
      } else {
        static_assert(false_constexpr<PointType>, "Unsupported Composite table!");
      }

      if constexpr (std::is_same_v<LSH, core::GaussianHashDense<CoordinateType, HashType>>) {
        partition_lshes.emplace_back(std::make_unique<LSH>(params_.dimension, par_para.k, par_para.l, params_.universe,
                                                           params_.seed ^ 93384688, par_para.bucket_width,
                                                           params_.bucket_id_width, params_.hash_table_width));
      } else if constexpr (std::is_same_v<LSH, core::CrossPolytopeHash2<CoordinateType, HashType>>) {
        assert(params_.num_rotations == NUM_ROTATIONS);
        partition_lshes.emplace_back(std::make_unique<LSH>(params_.dimension, par_para.k, par_para.l,
                                                           params_.seed ^ 93384688, params_.hash_table_width));
      } else if constexpr (std::is_same_v<LSH, core::AXequalYHash<CoordinateType, HashType>>) {
        // auto precomputed = LSH::read_precomputed(params_.eigen_filename, par_para.k*par_para.l,
        //                 find_next_power_of_two(params_.dimension), params_.fast_rotation));
        // auto hyperplanes = read_hyperplanes();
        partition_lshes.emplace_back(std::make_unique<LSH>(params_.dimension, par_para.k, par_para.l, params_.num_rotations,
                                                           params_.seed ^ 93384688, par_para.bucket_width,
                                                           params_.hash_table_width, params_.dim_Arows,
                                                           params_.second_step, params_.fast_rotation));
      } else {
        static_assert(false_constexpr<PointType>, "Unsupported LSH Function!");
      }
    }

    if constexpr (std::is_same_v<LSHTableType, core::StaticPartitionLSHTable<PointType, KeyType, LSH, HashType,
                  CompositeTable, MultiProbe, PartitionMetric, DataStorageType>>) { /*
                  auto precomputed = LSH::read_precomputed(params_.eigen_filename, params_.num_hash_funcs, params_.num_rotations,
                                     params_.rotation_dim, params_.dim_Acols,
                                    params_.fast_rotation);
                  auto pre_hash = std::make_unique<LSH::PreHasher>(std::move(precomputed), params_.num_hash_funcs, params_.dimension,
                                        params_.dim_Arows, params_.second_step, params_.fast_rotation,
                                        params_.num_rotations, params_.seed ^ 93384688);
                  ptrs_sanity_check(pre_hash);
                  lsh_table = std::make_unique<LSHTableType>(partition_lshes, partition_tables, *data_storage_,
                                     params_.num_setup_threads, params_.load_index,
                                     params_.index_path + params_.index_filename, pre_hash);*/
    } else if constexpr (std::is_same_v<LSHTableType, core::StaticLSHTable2<PointType, KeyType, LSH, HashType,
                         CompositeTable, MultiProbe, DataStorageType>>) {
      lsh_table = std::make_unique<LSHTableType>(partition_lshes[0].get(), partition_tables[0].get(), *data_storage_,
                                                 params_.num_setup_threads, params_.save_index,
                                                 params_.index_filename);
    } else {
      static_assert(false_constexpr<PointType>, "Unsupported LSH Table!");
    }

    ptrs_sanity_check(factory);
    ptrs_sanity_check(lsh_table);
    ptrs_sanity_check(data_storage_);
    assert(params_.num_partitions == static_cast<int_fast32_t>(partition_tables.size()));
    assert(params_.num_partitions == static_cast<int_fast32_t>(partition_lshes.size()));
    for (const auto& p : partition_tables)
      ptrs_sanity_check(p);
    for (const auto& p : partition_lshes)
      ptrs_sanity_check(p);

    table_.reset(new LSHNNTableWrapper<PointType, KeyType, ScalarType,
                 DistanceFunc, LSHTableType,
                 LSH, typename HashTable::Factory,
                 CompositeTable, DataStorageType>(
                   std::move(partition_lshes), std::move(lsh_table), std::move(factory),
                   std::move(partition_tables), std::move(data_storage_)));

    return std::move(table_);
  }

private:
  const PointSet& points_;
  const LSHConstructionParameters& params_;
  std::unique_ptr<DataStorageType> data_storage_;
  int_fast32_t num_bits_;
  int_fast64_t n_;
  std::unique_ptr<LSHNearestNeighborTable<PointType, KeyType>> table_ = nullptr;
};

} // namespace wrapper
} // namespace falconn

namespace falconn {

template <typename PointType, typename KeyType, typename PointSet>
std::unique_ptr<LSHNearestNeighborTable<PointType, KeyType>> construct_table(
  const PointSet& points, const LSHConstructionParameters& params) {
  wrapper::StaticTableFactory<PointType, KeyType, PointSet> factory(points,
                                                                    params);
  return factory.setup();
}

template <typename PointSet>
CoordinateType maximum_partition_metric(const PointSet& points) {
  CoordinateType val = 0.0;
  for (const auto& pt : points) {
    auto pt_metric = PartitionMetric::eval(pt);
    if (pt_metric > val) {
      val = pt_metric;
    }
  }
  return val;
}

} // namespace falconn

#endif
