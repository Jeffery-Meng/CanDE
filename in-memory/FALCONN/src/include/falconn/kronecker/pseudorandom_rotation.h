#ifndef __PSEUDORANDOM_ROTATION_H__
#define __PSEUDORANDOM_ROTATION_H__

#include <cassert>
#include <cmath>
#include <functional>
#include <random>
#include <type_traits>
#include <vector>

#include "fht.h"
#include "Eigen/Dense"

#include "global.h"
#include "utils.h"

namespace ONIAK {

// A class that performs pseudorandom rotation to fixed-length vectors using the HD3 algorithm.
// The rotation is fixed upon construction.
class PseudorandomRotation {
 public:
  PseudorandomRotation(int dim, int num_rotations, std::mt19937& rng):
      dim_(dim), num_rotations_(num_rotations) {
    fht_ = get_fht<DType>();
    assert(IsPowerOfTwo(dim) && "dim must be a power of 2");

    std::bernoulli_distribution bernoulli(0.5);
    diagonal_.resize(num_rotations, dim);
    const DType one_over_sqrt_dim = 1.0 / sqrt(dim);
    for (DType& val : diagonal_.reshaped()) {
      val = bernoulli(rng)? one_over_sqrt_dim: -one_over_sqrt_dim;
    }
  }

  // Performs rotation in-place for vector.
  void operator()(DRowVector& data) const;

  DRowVector operator()(const DRowVector& data) const;

 private:
  template<typename DType>
  static std::function<int(DType*, int)> get_fht() {
    if constexpr(std::is_same_v<DType, float>) {
        return fht_float;
    } else {
        return fht_double;
    }
  }
  // Dimension of vectors to rotate.
  int dim_;

  // Number of HD3 rotations
  int num_rotations_;

  // +/- 1 diagonal entries in D-matrices
  DMatrix diagonal_;

  // function that performs fast Hadamard transform.
  std::function<int(DType*, int)> fht_;
};

}  // namespace ONIAK

#endif // __PSEUDORANDOM_ROTATION_H__