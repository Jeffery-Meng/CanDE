#include "core/polytope_hash2.h"
#include "kronecker/kronecker_lsh.h"
#include "core/euclidean_distance.h"
#include "core/data_storage.h"
#include "core/multiprobe_cp.h"
#include "core/matrix_inner_product.h"
#include "simulation/data_generator.h"
#include "stop_watch.hpp"
#include <random>
#include <iostream>

using namespace falconn;
using namespace falconn::core;
using namespace ONIAK;
using namespace std;

constexpr int NUM_EXPERIMENTS = 1000000;
constexpr int DIMENSION = 128;

int main(int argc, char * argv[]) {
  std::random_device rd;
  uint32_t seed_alt;
  if (argc > 1) seed_alt = atoi(argv[1]);
  uint32_t seed  = rd();
  if (argc > 1) seed = seed_alt;
  std::mt19937 rng(seed);

  std::vector<VectorType> data_vecs;
  data_vecs.reserve(NUM_EXPERIMENTS);
  for (int i = 0; i < NUM_EXPERIMENTS; ++i) {
    data_vecs.push_back(UniformRandomVector(DIMENSION, rng).transpose());
  }
  
  ArrayDataStorage<VectorType> data_array(data_vecs);
  MatrixInnerProduct<VectorType, VectorPairType> distance;
  EuclideanDistanceDense euclidean;
  VectorPairType query = std::make_pair(UniformRandomVector(DIMENSION, rng).transpose(),
                                        UniformRandomVector(DIMENSION, rng).transpose());
  
  double sum = 0;
  StopWatch timer;
  typename ArrayDataStorage<VectorType>::FullSequenceIterator iter =
        data_array.get_full_sequence();
  while (iter.is_valid()) {
        auto cur_distance = distance(query, iter.get_point());
        sum += cur_distance;
        ++iter;
      }
  cout << timer.peek() << endl;
  timer.reset_and_start();
  iter =  data_array.get_full_sequence();
  while (iter.is_valid()) {
        auto cur_distance = euclidean(query.first, iter.get_point());
        sum += cur_distance;
        ++iter;
      }
      cout << timer.peek() << endl;
  cout << "Test passed." << endl;
  return 0;
}