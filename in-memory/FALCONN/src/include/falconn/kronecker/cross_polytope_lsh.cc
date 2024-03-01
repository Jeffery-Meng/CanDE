#include "cross_polytope_lsh.h"
#include <cmath>

namespace ONIAK {

int CrossPolytopeLSH::operator()(const DRowVector& data) const {
    DRowVector rotated_data = rotation_(data);
    int max_position = std::max_element(rotated_data.begin(), rotated_data.end(),
           [&](auto i, auto j){ return std::abs(i) < std::abs(j);}) - rotated_data.begin();
    return (rotated_data[max_position] > 0)? max_position + 1: -max_position - 1;  // positive 0, 1, 2, negative -1, -2, -3
}

DRowVector CrossPolytopeLSH::rotate(const DRowVector& data) const {
    return rotation_(data);
}

}  // namespace ONIAK