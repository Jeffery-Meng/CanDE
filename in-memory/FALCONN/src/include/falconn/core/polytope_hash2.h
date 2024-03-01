#ifndef __POLYTOPE_HASH2_H__
#define __POLYTOPE_HASH2_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>
#include <utility>

#include <Eigen/Dense>
#include <wyhash32.h>

#include "../ffht/fht_header_only.h"
#include "data_storage.h"
#include "heap.h"
#include "incremental_sorter.h"
#include "lsh_function_helpers.h"
#include "math_helpers.h"
#include "../kronecker/kronecker_lsh.h"

/* This file implements Kronecker-based Cross-Polytope Hashing for xy^T. */

namespace falconn {
namespace core {

template <typename CoordinateT = float, typename HashT = uint32_t>
class CrossPolytopeHash2 {
 private:
  class MultiProbeLookup;

 public:
  typedef HashT HashType;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      RotatedVectorType;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
      MatrixType;
  typedef std::pair<RotatedVectorType, RotatedVectorType> VectorPair;
  typedef std::vector<RotatedVectorType> TransformedVectorType;
  typedef std::vector<std::vector<ONIAK::MultiprobeType>> MultiprobeType;

  typedef VectorPair QueryType;
  // alternate query type.
  typedef MatrixType QueryType2;

  class HashTransformation {
   public:
    HashTransformation(const CrossPolytopeHash2& parent)
        : parent_(parent) {}

    // vector pair or matrix queries
    template <typename PointType>
    void apply(const PointType& v, MultiprobeType* result, int T) {
      assert(result != nullptr);
      result->resize(parent_.l_ * parent_.k_);
      for (int ii = 0; ii < parent_.l_ * parent_.k_; ++ii) {
        (*result)[ii] = parent_.kronecker_lshes_[ii].multiprobe(v, T);
      }
    }

    int get_l() const {return parent_.l_; }

   private:
    const CrossPolytopeHash2& parent_;
  };

  /* template base*/
  // Without Dummy, the code compiles in Clang, but not GCC 
  template <typename Dummy, typename BatchVectorType>
  class BatchHash {};

  // For hashing vector self-Kroneckers like xx^T
  template <typename Dummy>
  class BatchHash <Dummy, ArrayDataStorage<VectorType, KeyType>> {
   using BatchVectorType = ArrayDataStorage<VectorType, KeyType>;
   public:
    BatchHash(const CrossPolytopeHash2& parent)
        : parent_(parent),
          tmp_vector_(parent.rotation_dim_),
          tmp_hash_vector_(parent.k_) {}

    void batch_hash_single_table(const BatchVectorType& points, int_fast32_t l,
                                 std::vector<HashType>* res) {
      int_fast64_t nn = points.size();
      if (static_cast<int_fast64_t>(res->size()) != nn) {
        res->resize(nn);
      }

      typename BatchVectorType::FullSequenceIterator iter =
          points.get_full_sequence();
      for (int_fast64_t ii = 0; ii < nn; ++ii) {
        (*res)[ii] = 0;
        int_fast32_t pattern = l * parent_.k_;
        parent_.embed(iter.get_point(),  &tmp_vector_);
        ++iter;

        for (int_fast32_t jj = 0; jj < parent_.k_; ++jj) {
          tmp_hash_vector_[jj] = 
          // overload for self-Kronecker
              parent_.kronecker_lshes_[pattern++](tmp_vector_);
        }

        (*res)[ii] = wyhash32(tmp_hash_vector_.data(), tmp_hash_vector_.size() * sizeof(int),
             parent_.wy_seed_);
        (*res)[ii] &= (1<<parent_.hash_table_width_)-1;
      }
    }
private:
    const CrossPolytopeHash2& parent_;
    RotatedVectorType tmp_vector_;
    std::vector<int> tmp_hash_vector_;
  };

  // TODO: Can the FHT rotations be faster by doing 8 rotations (AVX float
  // register size) in parallel? If so, the table setup time could be made
  // faster by using a batch FHT below.
  //
  // For hashing vector pairs like xy^T.
  template <typename Dummy>
  class BatchHash <Dummy, ArrayDataStorage<VectorPairType, KeyType>> {
   using BatchVectorType = ArrayDataStorage<VectorPairType, KeyType>;
   public:
    BatchHash(const CrossPolytopeHash2& parent)
        : parent_(parent),
          tmp_vector_left_(parent.rotation_dim_),
          tmp_vector_right_(parent.rotation_dim_),
          tmp_hash_vector_(parent.k_) {}

    void batch_hash_single_table(const BatchVectorType& points, int_fast32_t l,
                                 std::vector<HashType>* res) {
      int_fast64_t nn = points.size();
      if (static_cast<int_fast64_t>(res->size()) != nn) {
        res->resize(nn);
      }

      typename BatchVectorType::FullSequenceIterator iter =
          points.get_full_sequence();
      for (int_fast64_t ii = 0; ii < nn; ++ii) {
        (*res)[ii] = 0;
        int_fast32_t pattern = l * parent_.k_;
        VectorPairType data_pair = iter.get_point();
        ++iter;
        parent_.embed(data_pair.first,  &tmp_vector_left_);
        parent_.embed(data_pair.second, &tmp_vector_right_);

        for (int_fast32_t jj = 0; jj < parent_.k_; ++jj) {
          tmp_hash_vector_[jj] = 
              parent_.kronecker_lshes_[pattern++](tmp_vector_left_, tmp_vector_right_);
        }

        (*res)[ii] = wyhash32(tmp_hash_vector_.data(), tmp_hash_vector_.size() * sizeof(int),
             parent_.wy_seed_);
        (*res)[ii] &= (1<<parent_.hash_table_width_)-1;
      }
    }

   private:
    const CrossPolytopeHash2& parent_;
    RotatedVectorType tmp_vector_left_, tmp_vector_right_;
    std::vector<int> tmp_hash_vector_;
  };

  int_fast32_t get_l() const { return l_; }
  int_fast32_t get_k() const { return k_; }
  int_fast32_t get_seed() const { return wy_seed_; }

  //  PointType can be pair or matrix
  template <typename PointType>
  void hash(const PointType& point, std::vector<HashType>* result) const {
    result->resize(l_);
    std::vector<int> tmp_hash(k_);
    size_t pattern = 0;
    for (int ll = 0; ll < l_; ++ll) {
      for (int kk = 0; kk < k_; ++kk) {
        tmp_hash[kk] = kronecker_lshes_[pattern++](point);
      }
      (*result)[ll] = wyhash32(tmp_hash.data(), tmp_hash.size() * sizeof(int), wy_seed_);
      (*result)[ll] &= (1<<hash_table_width_)-1;
    }
  }

  int_fast32_t hash_table_width() const {return hash_table_width_;}

  CrossPolytopeHash2(int_fast32_t rotation_dim, int_fast32_t k,
                        int_fast32_t l,
                        uint_fast64_t seed, int_fast32_t hash_table_width)
      : rotation_dim_(rotation_dim),
        k_(k),
        hash_table_width_(hash_table_width), l_(l),
        gen_(seed), wy_seed_(gen_()), seed_(seed) {
    if (rotation_dim_ < 1) {
      throw LSHFunctionError("Rotation dimension must be at least 1.");
    }

    if (k_ < 1) {
      throw LSHFunctionError(
          "Number of hash functions must be"
          "at least 1.");
    }

    if (l_ < 1) {
      throw LSHFunctionError("Number of hash tables must be at least 1.");
    }

    // http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    bool rotation_dim_power_of_two = !(rotation_dim & (rotation_dim - 1));
    if (!rotation_dim_power_of_two) {
      throw LSHFunctionError("Rotation dimension must be a power of two.");
    }   

    kronecker_lshes_.reserve(k_ * l_);
    for (int ii = 0; ii < k_ * l_; ++ii) {
      kronecker_lshes_.emplace_back(rotation_dim_, gen_);
    }
  }

 private:
  const int_fast32_t rotation_dim_;  // dimension of the vectors to be rotated
  const int_fast32_t k_, hash_table_width_;
  int_fast32_t l_;
  std::mt19937 gen_;
  uint32_t wy_seed_;
  const uint_fast64_t seed_;
  std::vector<ONIAK::KroneckerLSH> kronecker_lshes_;


  // Patch 0 after v so that its dimension becomes rotation_dim
  void embed(const RotatedVectorType& v, RotatedVectorType* result) const {
    // TODO: use something more low-level here?
    size_t vector_dim = v.size();
    for (size_t ii = 0; ii < vector_dim; ++ii) {
      (*result)[ii] = v[ii];
    }
    for (int_fast32_t ii = vector_dim; ii < this->rotation_dim_; ++ii) {
      (*result)[ii] = 0.0;
    }
  }
};

}  // namespace core
}  // namespace falconn

#endif
