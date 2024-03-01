#include "core/polytope_hash2.h"
#include "kronecker/kronecker_lsh.h"
#include "core/data_storage.h"
#include "core/multiprobe_cp.h"
#include "simulation/data_generator.h"
#include "stop_watch.hpp"
#include <random>
#include <iostream>

using namespace falconn;
using namespace falconn::core;
using namespace ONIAK;
using namespace std;

constexpr int NUM_EXPERIMENTS = 10000;
constexpr int DIMENSION = 128;
constexpr int PROBES = 100;
constexpr int L = 20;
constexpr int K = 4;

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
  
  CrossPolytopeHash2 cp_hash(DIMENSION, K, L, rng(), /*hash width*/ 20);
  ArrayDataStorage<VectorType> data_array(data_vecs);
  typename CrossPolytopeHash2<>::template BatchHash<ArrayDataStorage<VectorType>> batch_hash(cp_hash);
  typename CrossPolytopeHash2<>::HashTransformation hash_tran(cp_hash);
  MultiProbeCP<CrossPolytopeHash2<>> multiprobe(cp_hash, PROBES);
  
  StopWatch timer;
  std::vector<HashType> hash1, hash2;
  for (int ll = 0; ll < L; ++ll) {
    batch_hash.batch_hash_single_table(data_array, ll, &hash1);
  }
  cout << timer.peek() << endl;
  
  std::vector<std::vector<HashType>> hashes = {hash1, hash2};
  using MultiprobeVector = std::vector<typename CrossPolytopeHash2<>::MultiprobeType>;
  timer.reset_and_start();
  MultiprobeVector mp_vec;
  for (int exp = 0; exp < 100; ++exp) {
    auto data = data_vecs[exp];
    hash_tran.apply(std::make_pair(data, data), &mp_vec, PROBES);
    continue;
    multiprobe.setup_probing(std::move(mp_vec), L*PROBES);
    HashType pb;
    int_fast32_t table;
    for (int probe = 0; probe < L*PROBES; ++probe) {
      multiprobe.get_next_probe(&pb, &table);
    }
  }
  cout << timer.peek() << endl;
  cout << "Test passed." << endl;
  return 0;
}