#ifndef __DATA_GENERATOR_H__
#define __DATA_GENERATOR_H__

#include <random>

#include "kronecker/global.h"
#include "Eigen/Dense"

namespace ONIAK {

// Generate a uniform random rotation dim x dim.
DMatrix UniformRandomRotation(int dim, std::mt19937& rng);


// Generate a uniform random vector on a unit sphere.
DRowVector UniformRandomVector(int dim, std::mt19937& rng);


// Generates two correlated random vectors that have the given cosine value.
std::pair<DRowVector, DRowVector> TwoRandomVectors(int dim, DType cosine, std::mt19937& rng);

// Generates a correlated random vectors to 'input' that have the given cosine value.
DRowVector CorrelatedVector(const DRowVector& vector1, DType cosine, std::mt19937& rng);

}

#endif  // __DATA_GENERATOR_H__