#ifndef __RANGE_SUMMABLE_H__
#define __RANGE_SUMMABLE_H__

#include <cstdint>
#include <ctime>
#include <iterator>
#include <memory>
#include <random>
#include <vector>
#include <limits>
#include<algorithm>

#include "Eigen/Dense"

#include "data_storage.h"
#include "lsh_query_new.h"
#include "multiprobe.h"
#include "gaussian_hash.h"
#include "dyatree.h"
#include "wyhash32.h"

#include <fstream>

  // Generate the hash_vector for one seed and length is universe
  // Currently use float type for simplicity
  // The function change for Gaussian
unsigned log2dis(int x) {
		 int power = 0;
		 while (x >>= 1) {
			 ++power;
		 }
		 return power;
	 }

namespace falconn {
namespace core {


template <typename CoordinateType = float, typename HashType = uint32_t>
class RangeSummableGaussian
    : public GaussianHashBase<
          RangeSummableGaussian<CoordinateType, HashType>,
          Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>,
          CoordinateType, HashType> {
 public:
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      DerivedVectorT;
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::ColMajor>
      MatrixType;

  const DerivedVectorT& get_translation() const {return translation_;}
// For convinience we add universe here, though not needed
  RangeSummableGaussian(int dim, int_fast32_t k, int_fast32_t l, int_fast32_t universe,
                      uint_fast64_t seed, float w, int_fast32_t id_width, int_fast32_t hash_table_width)
      :GaussianHashBase<RangeSummableGaussian<CoordinateType, HashType>,
                           DerivedVectorT, CoordinateType, HashType>(dim, k, l, w, id_width, hash_table_width), 
      seed_(seed), gen_(seed), universe_(universe), cur_table_(-1){
    hash_seed_ = gen_();
    seed_hash2_ = gen_();

    //std::cauchy_distribution<CoordinateType> cauchy(0.0, 1.0); 
    std::uniform_real_distribution<CoordinateType> uniform_dist(0.0, this->bucket_width_);
// dim: data dimension
    //hyperplanes_.resize(this->k_ * this->l_, this->dim_);
    translation_.resize(this->k_ * this->l_);
    dya_trees_.reserve(l * k * dim);

    for (int ll =0; ll < this->l_; ++ll){
      for (int jj = ll * this->k_; jj < (ll+1) * this->k_; ++jj){
        translation_(jj) = uniform_dist(gen_);
      }
      
      for (int ii = ll * this->k_ * this->dim_; ii < (ll+1)* this->k_ * this->dim_; ++ii){
        dya_trees_.emplace_back(log2dis(universe), universe/2, gen_); 
          // L hash tables * K hash functions per table * D input dimenstions  log2dis(universe)
      }
    }

    
    //index_hash_.open("index_hash.txt");
    
  }
// (Ax+b)/w
  void get_multiplied_vector_all_tables(const DerivedVectorT& point,
                                        DerivedVectorT* res) const {
                                          
    //static std::ofstream probe_hash_("probe_hash.txt");
    int dst_idx = 0, re_idx = 0;
    for (int li = 0; li < this->l_; ++li){
      for (int ki = 0; ki < this->k_; ++ki){
        float result = 0.f;
        for (int dj = 0; dj < this->dim_; ++dj){
          unsigned idx = (unsigned) (point[dj] / 2);
          result += dya_trees_[dst_idx++].range_sum(idx);
        }
        result = (result + translation_[re_idx]) / this->bucket_width_;
        (*res)[re_idx++] = result;

        //probe_hash_ << result << "\t";
      }
      //probe_hash_ << std::endl;
    }
  }

  void get_multiplied_vector_single_table(const DerivedVectorT& point,
                                          int_fast32_t l,
                                          DerivedVectorT* res) {
    if (l != cur_table_){
      precomputes_.clear();

      int dst_idx = l*this->k_*this->dim_;
      for (; dst_idx < (l+1)* this->k_ * this->dim_; ++dst_idx) {
        precomputes_.push_back(dya_trees_[dst_idx].precompute());
      }
      cur_table_ = l;
    }

    int prec_idx = 0, tran_idx = l * this->k_;
    for (int ki = 0; ki < this->k_; ++ki){
      float result = 0.f;
      for (int dj = 0; dj < this->dim_; ++dj){
        unsigned idx = (unsigned) (point[dj] / 2);
        result += precomputes_[prec_idx++][idx];
      }
      result = (result + translation_[tran_idx++]) / this->bucket_width_;
      (*res)[ki] = result;
      //index_hash_ << result << "\t";
      
    }
    //index_hash_ << std::endl;
  }

  void hash(const DerivedVectorT& point, std::vector<HashType>* result,
            DerivedVectorT* tmp_hash_vector = nullptr) const {
    bool allocated = false;
    if (tmp_hash_vector == nullptr) {
      allocated = true;
      tmp_hash_vector = new DerivedVectorT(this->k_ * this->l_);
    }

    get_multiplied_vector_all_tables(point, tmp_hash_vector);

    std::vector<HashType>& res = *result;
    std::vector<int> res_rounded(this->k_, 0);
    if (res.size() != static_cast<size_t>(this->l_)) {
      res.resize(this->l_);
    }
    for (int_fast32_t ii = 0; ii < this->l_; ++ii) {
      for (int_fast32_t jj = 0; jj < this->k_; ++jj) {
        res_rounded[jj] =  hash_round((*tmp_hash_vector)[ii * this->k_ + jj], this->bucket_id_width_);
      }
      res[ii] = wyhash32(res_rounded.data(), this->k_ * sizeof(int), seed_hash2_);
      res[ii] &= (1<<this->hash_width_)-1;
    }

    if (allocated) {
      delete tmp_hash_vector;
    }
  }

  void hash_to_bucket(const DerivedVectorT& hash_vec, std::vector<HashType>& res) const{
    if (res.size() != static_cast<size_t>(this->l_)) {
      res.resize(this->l_);
    }
    std::vector<int> res_rounded(this->k_, 0);
    for (int_fast32_t ii = 0; ii < this->l_; ++ii) {
      for (int_fast32_t jj = 0; jj < this->k_; ++jj) {
        res_rounded[jj] = static_cast<int_fast32_t>(std::floor(hash_vec[ii * this->k_ + jj]));
      }
      res[ii] = wyhash32(res_rounded.data(), this->k_ * sizeof(int), seed_hash2_);
      res[ii] &= (1<<this->hash_width_)-1;
    }
  }

  void clear_precomputes(){
    precomputes_.clear();
  }

public:
  uint_fast32_t hash_seed_, seed_hash2_; // multiprobe also needs this

private:
  // ToW hashed Matrix A
  std::vector<DyaSimTree<unsigned>> dya_trees_;
      // uniform vector b
  DerivedVectorT translation_;
  std::mt19937 gen_; // Put seed inside? Probably?
  uint_fast64_t seed_;
  int_fast32_t universe_;

  int cur_table_; // current table being indexed
  std::vector<std::vector<float> > precomputes_;
 // std::ofstream index_hash_;
};

     

}  // namespace core
}  // namespace falconn

#endif
