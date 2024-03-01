#ifndef _CROSS_POLYTOPE_LSH_H__
#define _CROSS_POLYTOPE_LSH_H__

#include "pseudorandom_rotation.h"

namespace ONIAK {

// A class that computes Cross-Polytope (CP) LSH values for MIPS (maximum inner product search)
// The hash randomness is fixed upon construction.
class CrossPolytopeLSH {
 public:
  CrossPolytopeLSH(int dim, std::mt19937& rng): rotation_(
    dim, falconn::NUM_ROTATIONS, rng) {}

  int operator()(const DRowVector& data) const;

  DRowVector rotate(const DRowVector& data) const;

 private:
  PseudorandomRotation rotation_;
};

}  // namespace ONIAK

#endif   // _CROSS_POLYTOPE_LSH_H__