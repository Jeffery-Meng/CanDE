#ifndef __ROUNDING_HEADER_HPP__
#define __ROUNDING_HEADER_HPP__

#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <type_traits>

class NaiveRounding {
public:
    typedef  float DataT;
    static DataT forward(float a) {return a;}
    static float backward(DataT a) {return a;}
};

class JXRounding {
private:
    static std::vector<size_t> overflow_;
    constexpr static float stepL = 32.;
    constexpr static float stepH = 8.;
    static size_t counter_;
public:
    typedef char DataT;
    static float scale;

    static DataT forward(float a) {
        int a_i = roundf(a * scale * stepL);
        if (abs(a_i) < 128) {
            return a_i; // return char
        } else { //overflow
            overflow_.push_back(counter_);
            int b_i = roundf(a * scale * stepH);
            if (b_i >=128) {
                return 127;
            } else if (b_i <=-128) {
                return -127;
            } else {
                return b_i;
            }
        }
        ++counter_;
    }
    static float backward(DataT a, bool overf = false) {
        float step = overf? stepH: stepL;
        return a / scale / step;
    }

    static std::vector<size_t>&& dump_overflow() {
        return std::move(overflow_);
    }

    static size_t of_size() {
        return overflow_.size();
    }
};

float JXRounding::scale = 1.0;
size_t JXRounding::counter_ = 0;
std::vector<size_t> JXRounding::overflow_;

class WHYRounding {
private:
public:
    typedef char DataT;
    static float scale;

    static DataT forward(float a) {
        int a_i = roundf(a * scale);
        if (a_i >=128) {
            return 127;
        } else if (a_i <=-128) {
            return -127;
        } else {
            return a_i;
        }
    }
    static float backward(DataT a) {
        return a / scale;
    }
};

float WHYRounding::scale = 16.;

template <int PartitionId>
class RoundedFloat {
private:
    char data_;
public:
    static float scale;
    RoundedFloat(float a) {
        int a_i = roundf(a * scale);
        if (a_i >=128) {
            data_ = 127;
        } else if (a_i <=-128) {
            data_ = -127;
        } else {
            data_ = a_i;
        }
    }

    operator float() const {
        return data_ / scale;
    }
};
template <int PartitionId>
float RoundedFloat<PartitionId>::scale = 16.;

///////////////////////////////////////////
template <typename RoundingT>
class RoundingTestbed {
private:
    static float lossy_compress(float a) {
        if constexpr (std::is_same_v<RoundingT, JXRounding>) {
            size_t last_sz = RoundingT::of_size();
            DataT d_cmp = RoundingT::forward(a);
            size_t cur_sz = RoundingT::of_size();
            return RoundingT::backward(d_cmp, last_sz != cur_sz);
        } else {
             return RoundingT::backward(RoundingT::forward(a));
        }
    }

    static double l2_norm(const std::vector<float> &vec) {
        double l2_squared = std::accumulate(vec.begin(), vec.end(), 0.0, 
        [](double a, float b) ->double {return a + b * b;});
        return sqrt(l2_squared);
    }

    std::mt19937 rng_;
public:
    RoundingTestbed(unsigned seed = 0x312f373d): rng_(seed) {}
    typedef typename RoundingT::DataT DataT;

    double sketch(float stddev, int size) {
        std::normal_distribution<float> gaussian(0, stddev);
        std::vector<float> array(size);
        for (auto& item:array) {
            item = lossy_compress(gaussian(rng_));
        }
        return l2_norm(array);
    }


};

#endif