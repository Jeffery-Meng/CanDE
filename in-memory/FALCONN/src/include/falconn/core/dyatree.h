#ifndef DYATREE_H
#define DYATREE_H
// #define DEBUG

#include <random>
#include <string>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <limits>
#include <numeric>
#include "wyhash32.h"
//#include <boost/align/aligned_allocator.hpp>

#include <chrono>


namespace falconn{
namespace core{

template <typename UniverseT = unsigned>
class DyaSimTree{
private:
    const unsigned _level;
    const UniverseT _universe;
    std::vector<unsigned> _seedl, _seedr;
    const unsigned _seed_root, _seed_w;
    static constexpr float sqrt2 = 1.4142135623730951;
    const float _denom;

public:
    DyaSimTree(unsigned level, UniverseT universe, std::mt19937& rng): 
        _level(level), _universe(universe), _seedl(level+1), _seedr(level+1), _seed_root(rng()), 
        _seed_w(rng()), _denom(std::pow(2.f, -(float)_level/2.0f))
    {
        assert(!std::numeric_limits<UniverseT>::is_signed); // only support unsigned type
        assert(level == 8 * sizeof(UniverseT) || (level < 8 * sizeof(UniverseT) && universe <= (1u << level)));

        for (int idx = 0; idx <= _level; ++idx){
            _seedl[idx] = rng();
            _seedr[idx] = rng();
        }

    }

    float range_sum(UniverseT x) const{
        // if (start < 0 || end >= _max || start > end) return false;
        if (x > _universe) return -1.f;
        float val = wy2gau(_seed_root) * (float)x * _denom; // root level
        return val + rs_recur(x, _seed_w, _level, _denom);
    }

    std::vector<float> precompute(){
        std::vector<float> result(_universe+1);
        float val = wy2gau(_seed_root) / _denom; // root level
        if (_level < 8 * sizeof(UniverseT) && _universe == 1 << _level){
            result[_universe] = val;
        }
        pc_recur(0, 0.f, val, _seed_w, _level, 1 / _denom / 2, result);
        result[0] = 0.f;
        return result;
    }

private:
    // input interval [start, end], outside interval size 1<<level, 
    float rs_recur(UniverseT x, unsigned hash, unsigned level, float denom) const{
        if (x == 0 || level == 0) return 0;

        float val;

        unsigned n = 1u<<(level-1);
        if (x>n){
            x = x - n; 
            val = wy2gau(hash) * (float)(n - x) * denom;
            val += rs_recur(x, wyhash32(&hash, sizeof(unsigned), _seedr[level]), level - 1, denom * sqrt2);
        } else {
            val = wy2gau(hash) * (float)x * denom;
            val += rs_recur(x, wyhash32(&hash, sizeof(unsigned), _seedl[level]), level - 1, denom * sqrt2);
        }
        return val;
    }

    void pc_recur(UniverseT x, float val, float z, unsigned hash, unsigned level, float scale, std::vector<float>& arr){
        if (level == 0) return;
        
        unsigned levelm1 = level - 1;
        float w = wy2gau(hash) * scale;
        float z1 = z / 2.f + w, z2 = z - z1, val2 = val + z1;
        UniverseT pos = x + (1u << levelm1);
        //if (level > 12) {
         //           std::cout << level << "\t" << pos << "\t" << val2 << std::endl;
        //}
        if (pos <= _universe){
            arr[pos] = val2;
            pc_recur(pos, val2, z2, wyhash32(&hash, sizeof(unsigned), _seedr[level]), level - 1, scale / sqrt2, arr);
        }
        pc_recur(x, val, z1, wyhash32(&hash, sizeof(unsigned), _seedl[level]), level - 1, scale / sqrt2, arr);
    }
};
}
} // namespace 
#endif 
