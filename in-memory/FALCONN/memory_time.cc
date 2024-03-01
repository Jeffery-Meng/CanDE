#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "Timer.hpp"

using namespace std;

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

// For cast, need to specialize the value of scale, default value set to 0
float RoundedFloat::scale = 16.;

double run_dim(int num_filter) {
    size_t n = 1e6;
    std::vector<std::vector<RoundedFloat>> filters(n);
    float cnt = 0;
    for (auto& vec : filters) {
        vec.reserve(num_filter);
        for (int i = 0; i < num_filter; ++i) {
            vec.push_back(cnt);
            cnt += 1;
        }
    }
    cout << "start measure" << endl;
    volatile float sum = 0;
    HighResolutionTimer timer;
    timer.restart();
    for (const auto& vec : filters) {
        for (const auto& val : vec) {
            sum += val;
        }
    }
    return timer.elapsed();
} 

int main() {
    std::cout << "start" << std::endl;
    constexpr int num_filter[] = {2048, 4096, 8192};
    for (int i = 0; i < 3; ++i) {
        std::cout << run_dim(num_filter[i]) << std::endl;
    }
    return 0;
}