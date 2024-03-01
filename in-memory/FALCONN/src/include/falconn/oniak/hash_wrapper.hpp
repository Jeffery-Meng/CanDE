#ifndef __HASH_WRAPPER_HPP__
#define __HASH_WRAPPER_HPP__

#include "wyhash32.h"
#include <vector>
namespace ONIAK {

class WYHash {
 public:
  WYHash(): mod_(0xffffffff), seed_(3000750715) {}
  WYHash(uint32_t seed, uint32_t mask):  mask_(mask), seed_(seed){}

  uint32_t operator()(int x) const {
    uint32_t result = wyhash32(&x, sizeof(int), seed_);
    return result & mask_;
  }

  uint32_t operator()(int x, int y) const {
    int xy[2];
    xy[0] = x;
    xy[1] = y;
    uint32_t result = wyhash32(xy, sizeof(int) * 2, seed_);
    return result & mask_;
  }

  uint32_t& seed() {return seed_;}
  uint32_t& mod() {return mod_;}
  uint32_t& mask() {return mask_;}

 private:
  uint32_t mask_;
  uint32_t mod_;
  uint32_t seed_;
};


}

#endif