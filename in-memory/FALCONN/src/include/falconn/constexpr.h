#ifndef __CONSTEXPR_H__
#define __CONSTEXPR_H__

namespace falconn {
/* Here are global settings */
// Please change these options before compilation.

constexpr bool kUsePartition = false;
constexpr bool kUseAlternateQueryType = false;
// whether the data are self-Kronecker products of vectors
constexpr bool kSelfKroneckerData = true;
constexpr float MARGIN = 1e-4;

constexpr int NUM_ROTATIONS = 3;

using KeyType = int32_t;
using CoordinateType = float;
typedef uint32_t HashType;

enum class LSHTypes {
  kGaussian, kCrosspolytope
};

constexpr LSHTypes lsh_type = LSHTypes::kGaussian;

using DoubleTriplet = std::tuple<double, double, double>;

}

#endif