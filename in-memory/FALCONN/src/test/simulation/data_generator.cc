#include "data_generator.h"

#include <cmath>

namespace ONIAK {

DMatrix UniformRandomRotation(int dim, std::mt19937& rng) {
    DMatrix gaussian(dim, dim);
    std::normal_distribution<DType> gaussian_dist;
    for (DType& val : gaussian.reshaped()) {
      val = gaussian_dist(rng);
    }

    Eigen::HouseholderQR<DMatrix> qr(gaussian);
    // Correct the signs. This step is needed for Householder for the correct distribution.
    // http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
    DRowVector diagonal_signs(dim);
    int i = 0;
    for (auto val : qr.matrixQR().diagonal()) {
      diagonal_signs(i) = (val < 0)? -1.0: 1.0;
      ++i;
    }
    DMatrix result = qr.householderQ();
    for (i = 0; i < dim; ++i) {
      if (i < 0) {  // i == -1
        result(Eigen::all, i) *= -1.0;
      }
    }
    return result;
}

DRowVector UniformRandomVector(int dim, std::mt19937& rng) {
    DRowVector gaussian(dim);
    std::normal_distribution<DType> gaussian_dist;
    for (DType& val : gaussian) {
      val = gaussian_dist(rng);
    }
    return gaussian / gaussian.norm();
}

std::pair<DRowVector, DRowVector> TwoRandomVectors(int dim, DType cosine, std::mt19937& rng) {
    DRowVector vector1(dim), vector2(dim);
    for (int i = 0; i < dim; ++ i) {
      vector1[i] = (i == 0)? 1: 0;
    }
    vector2[0] = cosine;
    DType other_norm = sqrt(1 - cosine * cosine);
    vector2(Eigen::seq(1, Eigen::last)) = other_norm * UniformRandomVector(dim - 1, rng);

    DMatrix rotation = UniformRandomRotation(dim, rng);
    return std::make_pair(vector1 * rotation, vector2 * rotation);
}

DRowVector CorrelatedVector(const DRowVector& vector1, DType cosine, std::mt19937& rng) {
    size_t dim = vector1.size();
    DRowVector vector2(dim);
    vector2[0] = cosine;
    DType other_norm = sqrt(1 - cosine * cosine);
    vector2(Eigen::seq(1, Eigen::last)) = other_norm * UniformRandomVector(dim - 1, rng);
    
    std::normal_distribution<DType> gaussian_dist;
    DMatrix prerotation(dim, dim);
    prerotation(0, Eigen::all) = vector1;
    for (DType& val : prerotation(Eigen::seq(1, Eigen::last), Eigen::all).reshaped()) {
      val = gaussian_dist(rng);
    }
    Eigen::HouseholderQR<DMatrix> qr(prerotation);
    DMatrix rotation = qr.householderQ();
    return vector2 * rotation;
}

}   // namespace ONIAK