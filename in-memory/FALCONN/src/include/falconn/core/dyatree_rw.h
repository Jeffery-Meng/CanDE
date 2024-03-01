#ifndef DYATREE_RW_H
#define DYATREE_RW_H
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
#include "wyhash/wyhash32.h"
//#include <boost/align/aligned_allocator.hpp>
#include "marsaglia/5tbl.h"

#include <chrono>


namespace falconn{
namespace core{

template <typename UniverseT = unsigned>
class DyaSimTreeRW{
private:
    const unsigned _level;
    const UniverseT _universe;
    std::vector<unsigned> _seedl, _seedr, _seed_rs;
    const unsigned _seed_w;
    int _sum_root;
    static constexpr float sqrt2 = 1.4142135623730951;
    static constexpr double log2 = 0.69314718055;
    static constexpr double twop32 = 4294967296;

    
public:
    static std::vector<MarsagliaGenerator> _randwalks;
    static std::vector<std::vector<MarsagliaGenerator>> _posteriors;
    static int num_samples;
    static int num_attempts;
    static unsigned table_size;
    static double prob_sum;
    DyaSimTreeRW(unsigned level, UniverseT universe, std::mt19937& rng): 
        _level(level), _universe(universe), _seedl(level+1), _seedr(level+1), _seed_rs(level+1),
        _seed_w(rng())
    {
        assert(!std::numeric_limits<UniverseT>::is_signed); // only support unsigned type
        assert(level == 8 * sizeof(UniverseT) || (level < 8 * sizeof(UniverseT) && universe <= (1u << level)));
        if (_randwalks.size() < _level + 1){
            MarsagliaGenerator::set_factable(1u << _level);
            _randwalks.reserve(_level+1);
        }
        for (int l = _randwalks.size(); l < _level + 1; ++l){
            _randwalks.emplace_back(1u << l, -1);
            if (l == 0 || l > 11) continue;
            _posteriors.emplace_back();
            int offset = _randwalks[l].offset, range_len = _randwalks[l].range_len();
            _posteriors[l-1].reserve(range_len);
            for (int z = offset; z < offset + range_len; ++z) {
                _posteriors[l-1].emplace_back(1u << (l-1), z);
                table_size += _posteriors[l-1][z-offset].len();
                //std::cout << _posteriors[l-1][z-offset].len() << "\t" <<  table_size  << std::endl;
            }
        }

        for (int idx = 0; idx <= _level; ++idx){
            _seedl[idx] = rng();
            _seedr[idx] = rng();
            _seed_rs[idx] = rng();
        }

        unsigned seed_root = rng();
        _sum_root = _randwalks[_level].Dran(seed_root);
    }

    int range_sum(UniverseT x) const{
        // if (start < 0 || end >= _max || start > end) return false;
        assert(x >= 0 && x < _universe);
        return 2 * rs_recur(x, _sum_root, _seed_w, _level) - x;
    }

    /*std::vector<float> precompute(){
        std::vector<float> result(_universe+1);
        float val = wy2gau(_seed_root) / _denom; // root level
        if (_level < 8 * sizeof(UniverseT) && _universe == 1 << _level){
            result[_universe] = val;
        }
        pc_recur(0, 0.f, val, _seed_w, _level, 1 / _denom / 2, result);
        result[0] = 0.f;
        return result;
    }*/

private:
    static double prob_calc(int n, int x, int z){
        int bias = (z-n)/2 - (z-n<0 && (z-n)%2!=0);
        int x0 = (z+1) / 2;
        double lnratio = MarsagliaGenerator::_factable.lncomb(n, x0) + MarsagliaGenerator::_factable.lncomb(n, z-x0) 
                         - MarsagliaGenerator::_factable.lncomb(n, x0-bias);
        double lnprob = MarsagliaGenerator::_factable.lncomb(n, x) + MarsagliaGenerator::_factable.lncomb(n, z-x) -
                         MarsagliaGenerator::_factable.lncomb(n, x-(z-n)/2);
        //std::cout << exp(lnprob+ n * log2 -  MarsagliaGenerator::_factable.lncomb(2*n, z)  )  << std::endl;//"\t" << lnratio << "\t" << (lnprob - lnratio) << std::endl;
        //std::cout << n << "\t" << x << "\t" << z << "\t" << x0 << "\t" << exp(lnprob - lnratio) << std::endl;
        //std::cout << exp(MarsagliaGenerator::_factable.lncomb(n, x0) + MarsagliaGenerator::_factable.lncomb(n, z-x0) 
         //          - MarsagliaGenerator::_factable.lncomb(n, x0-bias) + n * log2 - 
        //           MarsagliaGenerator::_factable.lncomb(2*n, z)  ) << std::endl;
        return exp(lnprob-lnratio);
    }

    int rej_sample(int z, unsigned level, unsigned hash, unsigned seed) const{
        if (z == 0) return 0;
        ++num_samples;

        int n = 1u << level;
        if (z==2*n) return n;
        int x = (z+1) / 2;
        int bias = (z-n)/2 - (z-n<0 && (z-n)%2!=0);
        //std::cout << z << "\t" << x << std::endl;
        //double lnratio = MarsagliaGenerator::_factable.lncomb(n, x) + MarsagliaGenerator::_factable.lncomb(n, z-x) 
          //               - MarsagliaGenerator::_factable.lncomb(n, x-bias);//+ n * log2- lncomb(2*n, z)
        double lnratio = MarsagliaGenerator::_factable.lncomb(n, x) + MarsagliaGenerator::_factable.lncomb(n, z-x) 
                         - MarsagliaGenerator::_factable.lncomb(n, x-bias);//+ n * log2- lncomb(2*n, z)
        
        x = _randwalks[level].Dran(hash) + bias;
        hash = wyhash32(&hash, sizeof(unsigned), seed);
        double prob = exp(MarsagliaGenerator::_factable.lncomb(n, x) + MarsagliaGenerator::_factable.lncomb(n, z-x) -
                         MarsagliaGenerator::_factable.lncomb(n, x-bias)  - lnratio);
        //x = _randwalks[level].Dran(hash) + bias;
        //hash = wyhash32(&hash, sizeof(unsigned), seed);
        //double prob = prob_calc(n,x,z);
        prob_sum += prob;
        //std::cout << prob << std::endl;
        //+ n * log2- lncomb(2*n, z) 
        //std::cout << z << "\t" << x << "\t" << n << "\t" << lnratio << "\t" << prob << std::endl;
        ++num_attempts;
        while(prob != 1. && hash > (unsigned)(prob * twop32)){
            //if (prob >= 1 || prob == 0)// || prob < 0.001)
            //std::cout << n << "\t" << z << "\t" << x << "\t" << prob << "\t" << bias<< std::endl;
            hash = wyhash32(&hash, sizeof(unsigned), seed);
            x = _randwalks[level].Dran(hash)+ bias;
            hash = wyhash32(&hash, sizeof(unsigned), seed);
            //prob = prob_calc(n,x,z);
            prob = exp(MarsagliaGenerator::_factable.lncomb(n, x) + MarsagliaGenerator::_factable.lncomb(n, z-x) 
                    - MarsagliaGenerator::_factable.lncomb(n, x-bias)  - lnratio);
            prob_sum += prob;
            ++num_attempts;
        }
        if (x < 0 or x > z)
        {
            std::cout << n << "\t" << z << "\t" << x << "\t" << prob << "\t" << bias<< std::endl;
            exit(1);
        }
        return x;
    }

    // input interval [start, end], outside interval size 1<<level, 
    int rs_recur(UniverseT x, int range_sum, unsigned hash, unsigned level) const{
        int n = 1u<<(level-1);
        if (x == 0 || x == 2 * n)  return 0;
        if (level == 0) return range_sum;

        int offset = _randwalks[level].offset, range_len = _randwalks[level].range_len();
        int first_half;
        //int first_half = rej_sample(range_sum, level-1, hash, _seed_rs[level]);
        if  (n > 1024 || range_sum - offset < 0 || range_sum - offset >= range_len){
            //std::cout << n << "\t" << x << "\t" << range_sum << "\t" << offset << "\t" << range_len << std::endl;
            first_half = rej_sample(range_sum, level-1, hash, _seed_rs[level]);
        } else first_half = _posteriors[level-1][range_sum-offset].Dran(hash);
        int second_half = range_sum - first_half;
        int val;
        //if (first_half < 0 || second_half < 0){
       //     std::cout << n << "\t" << range_sum <<"\t"<< offset << "\t" << range_len << "\t" << first_half << "\t" << second_half << std::endl;
       // }

        //std::cout << n << "\t" << x << "\t" << range_sum << std::endl;
        if (x>=n){
            x = x - n; 
            val = first_half + rs_recur(x, second_half, wyhash32(&hash, sizeof(unsigned), _seedr[level]), level - 1);
        } else {
            val = rs_recur(x, first_half, wyhash32(&hash, sizeof(unsigned), _seedl[level]), level - 1);
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

template<typename UniverseT>
int DyaSimTreeRW<UniverseT>::num_samples = 0;

template<typename UniverseT>
int DyaSimTreeRW<UniverseT>::num_attempts = 0;

template<typename UniverseT>
unsigned DyaSimTreeRW<UniverseT>::table_size = 0;

template<typename UniverseT>
double DyaSimTreeRW<UniverseT>::prob_sum = 0.;

template<typename UniverseT>
std::vector<MarsagliaGenerator> DyaSimTreeRW<UniverseT>::_randwalks;
template<typename UniverseT>
std::vector<std::vector<MarsagliaGenerator>> DyaSimTreeRW<UniverseT>::_posteriors;
}
} // namespace 
#endif 
