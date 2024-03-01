#include "kronecker_lsh.h"

#include <cmath>
#include <numeric>
#include <functional>
#include "utils.h"
#include "../core/heap.h"
#include <utility>
#include <cassert>
#include <vector>

namespace ONIAK {

using Heap = falconn::core::SimpleHeap<DType, std::pair<int, int>>;

int KroneckerLSH::kronecker_value(int hash_val1, int hash_val2) const {
    int hash_v = (std::abs(hash_val1) -1) * dim_ +  (std::abs(hash_val2) -1);
    bool bit1 = std::signbit(hash_val1);
    bool bit2 = std::signbit(hash_val2);
    bool bit = bit1 ^ bit2;
    return bit? hash_v: -hash_v;
}


int KroneckerLSH::operator()(const DRowVector& data) const {
    assert(data.size() == dim_);
    int hash_val1 = cp_lsh1_(data);
    int hash_val2 = cp_lsh2_(data);
    return kronecker_value(hash_val1, hash_val2);
}

int KroneckerLSH::operator()(const DRowVector& dleft, const DRowVector& dright) const {
    assert(dleft.size() == dim_ && dright.size() == dim_);
    int hash_val1 = cp_lsh1_(dleft);
    int hash_val2 = cp_lsh2_(dright);
    return kronecker_value(hash_val1, hash_val2);
}

std::vector<MultiprobeType> KroneckerLSH::multiprobe(const DMatrix& matrix, int T) const {
    assert(matrix.rows() == dim_ && matrix.cols() == dim_);
    DMatrix rotated = matrix;
    // row_wise rotation
    for (int rr = 0; rr < dim_; ++rr) {
        rotated(rr, Eigen::all) = cp_lsh2_.rotate(rotated(rr, Eigen::all));
    }
    rotated = rotated.transpose();
    for (int rr = 0; rr < dim_; ++rr) {
        rotated(rr, Eigen::all) = cp_lsh1_.rotate(rotated(rr, Eigen::all));
    }
    rotated = rotated.transpose();
    auto rotated_view = rotated.reshaped();

    std::vector<MultiprobeType> vals;
    vals.reserve(dim_ * dim_);
    for (int dd = 0; dd < dim_ * dim_; ++dd) {
        DType score = rotated_view[dd];
        vals.emplace_back((score > 0)? dd: -dd, std::abs(score));
    }
    auto lambda = [&](auto i, auto j) {return i.score > j.score; };
    std::nth_element(vals.begin(), vals.begin() + T, vals.end(), lambda);
    vals.resize(T);
    std::sort(vals.begin(), vals.end(), lambda);
    DType best = vals[0].score;
    for (MultiprobeType& mp : vals) {
        mp.score = (best - mp.score) * (best - mp.score);
    }
    return vals;
}

std::vector<MultiprobeType> KroneckerLSH::multiprobe(const DRowVector& dleft,
 const DRowVector& dright, int T) const {
    DRowVector rot1 = cp_lsh1_.rotate(dleft);
    DRowVector rot2 = cp_lsh2_.rotate(dright);
    DRowVector rot1_abs = rot1.cwiseAbs();
    DRowVector rot2_abs = rot2.cwiseAbs();    
    auto order1 = Orders(rot1_abs.begin(), rot1_abs.end(), 
        [&](int i, int j) {return rot1_abs(i) > rot1_abs(j);} );
    auto order2 = Orders(rot2_abs.begin(), rot2_abs.end(), 
        [&](int i, int j) {return rot2_abs(i) > rot2_abs(j);} );  
    Heap heap;
    
    std::vector<MultiprobeType> vals;
    vals.reserve(T);
    auto value_func = [&] (int t1, int t2) { return std::abs(rot1(order1[t1]) * rot2(order2[t2]));};
    auto index_func = [&] (int t1, int t2) {
        int v1 = (rot1(order1[t1]) > 0)? order1[t1] + 1: -order1[t1] - 1;
        int v2 = (rot2(order2[t2]) > 0)? order2[t2] + 1: -order2[t2] - 1;
        return kronecker_value(v1, v2);
    };
    heap.insert(-value_func(0, 0), std::make_pair(0, 0));
    while(vals.size() < static_cast<size_t>(T) && !heap.empty()) {
        std::pair<int, int> key;
        DType data;
        heap.extract_min(&data, &key);
        vals.emplace_back(index_func(key.first, key.second), -data);

        if (key.first == 0 && key.second < dim_ - 1) {
            heap.insert(-value_func(0, key.second + 1), std::make_pair(0, key.second + 1));
        }
        if (key.first < dim_ - 1) {
            heap.insert(-value_func(key.first + 1, key.second), std::make_pair(key.first + 1, key.second));
        }      
    }
    
    DType best = vals[0].score;
    for (MultiprobeType& mp : vals) {
        mp.score = (best - mp.score) * (best - mp.score);
    }
    return vals;
}

/* used for debugging */
std::vector<MultiprobeType> KroneckerLSH::multiprobe_slow(const DRowVector& dleft,
 const DRowVector& dright, int T) const {
    DRowVector rot1 = cp_lsh1_.rotate(dleft);
    DRowVector rot2 = cp_lsh2_.rotate(dright);
    std::vector<MultiprobeType> vals;

    vals.reserve(dim_ * dim_);
    for (int ii = 0; ii < dim_; ++ii) {
        int v1 = (rot1(ii) > 0)? ii + 1: -ii - 1;
        for (int jj = 0; jj < dim_; ++ jj) {
            int v2 = (rot2(jj) > 0)? jj + 1: -jj - 1;
            int idx = kronecker_value(v1, v2);
            vals.emplace_back(idx, std::abs(rot1(ii) * rot2(jj)));
        }
    }
    std::sort(vals.begin(), vals.end());
    vals.resize(T);
    DType best = vals[0].score;
    for (MultiprobeType& mp : vals) {
        mp.score = (best - mp.score) * (best - mp.score);
    }
    return vals;
}


}  // namespace ONIAK