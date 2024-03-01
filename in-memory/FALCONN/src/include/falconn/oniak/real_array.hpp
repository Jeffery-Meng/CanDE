#ifndef __REAL_ARRAY_HPP__
#define __REAL_ARRAY_HPP__

#include "../fileio.h"
#include <string>
#include <vector>

// A simple array structure indexed by real numbers.
// Provides simple I/O support

namespace ONIAK {

class RealArray {
public:

RealArray(double start, double end, double step, std::vector<float> array = {}) :
start_(start), end_(end), step_(step), array_(std::move(array)) {}

// can be copied
RealArray(const RealArray& other) = default;

explicit RealArray(std::string filename) {
    auto buffer = falconn::read_data<std::vector<float>>(filename);
    assert(buffer.size() >= 2 && buffer[0].size() >= 3);
    start_ = buffer[0][0];
    end_ = buffer[0][1];
    step_ = buffer[0][2];
    array_ = std::move(buffer[1]);
}

float operator[](double x) {
    int idx = (x - start_) / step_;
    if (idx < 0) idx = 0;
    else if (idx >= static_cast<int>(array_.size())) idx = array_.size() - 1;
    return array_[idx];
}

std::vector<float>& array() {return array_;}
double& start() {return start_;}
double& end() {return end_;}
double& step() {return step_;}

private:
double start_, end_, step_;
std::vector<float> array_;
};

}

#endif