#include "pseudorandom_rotation.h"

namespace ONIAK {

void PseudorandomRotation::operator()(DRowVector& data) const {
    assert(data.size() == dim_ && "data must have the same size as dim");
    const int log_dim = LogOfTwo(dim_);

    for (int_fast32_t rotation = 0; rotation < num_rotations_; ++rotation) {
        data = data.cwiseProduct(diagonal_.row(rotation));
        fht_(data.data(), log_dim);
    }
}

DRowVector PseudorandomRotation::operator()(const DRowVector& data) const {
    assert(data.size() == dim_ && "data must have the same size as dim");
    const int log_dim = LogOfTwo(dim_);
    DRowVector result;

    for (int_fast32_t rotation = 0; rotation < num_rotations_; ++rotation) {
        if (rotation == 0) {
        result = data.cwiseProduct(diagonal_.row(rotation));
        } else {
        result = result.cwiseProduct(diagonal_.row(rotation));
        }
        fht_(result.data(), log_dim);
    }
    return result;
}

}  // namespace ONIAK
