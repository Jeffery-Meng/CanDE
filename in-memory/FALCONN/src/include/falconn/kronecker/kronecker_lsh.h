#ifndef _KRONECKER_LSH_H__
#define _KRONECKER_LSH_H__

#include "cross_polytope_lsh.h"
#include "pseudorandom_rotation.h"

#include "global.h"

namespace ONIAK {

struct MultiprobeType {
int index;
DType score;
MultiprobeType(int i, DType s): index(i), score(s) {}
MultiprobeType() = default;
/* sort from greatest to smallest, so use inverse ordering.*/
bool operator<(const MultiprobeType& other) const { return score > other.score; }
bool operator==(const MultiprobeType& other) const {
 return std::abs(score-other.score) < 1e-8;
}
};

// A class that computes Cross-Polytope (CP) LSH values for MIPS (maximum inner product search)
// The hash randomness is fixed upon construction.
class KroneckerLSH {
 public:

  KroneckerLSH(int dim, std::mt19937& rng): dim_(dim), cp_lsh1_(
    dim, rng), cp_lsh2_(dim, rng) {}
  
  int operator()(const DRowVector& data) const;
  int operator()(const DRowVector& dleft, const DRowVector& dright) const;
  int operator()(const std::pair<DRowVector, DRowVector>& vector_pair) const {
    return this->operator()(vector_pair.first, vector_pair.second);
  }
  int operator()(const DMatrix& matrix) {
    std::vector<MultiprobeType> result = multiprobe(matrix, 1);
    return result[0].index;
  }

  std::vector<MultiprobeType> multiprobe(const DRowVector& dleft, const DRowVector& dright, int T) const;
  std::vector<MultiprobeType> multiprobe(const std::pair<DRowVector, DRowVector>& vector_pair, int T) const {
    return multiprobe(vector_pair.first, vector_pair.second, T);
  };
  std::vector<MultiprobeType> multiprobe(const DMatrix& matrix, int T) const;

  std::vector<MultiprobeType> multiprobe_slow(const DRowVector& dleft, const DRowVector& dright, int T) const;

 private:
  int dim_;
  CrossPolytopeLSH cp_lsh1_, cp_lsh2_;
  int kronecker_value(int hash_val1, int hash_val2) const;

  
};

}  // namespace ONIAK

#endif   // _KRONECKER_LSH_H__