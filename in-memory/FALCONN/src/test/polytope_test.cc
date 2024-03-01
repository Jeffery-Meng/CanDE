#include "core/polytope_hash2.h"
#include "kronecker/kronecker_lsh.h"
#include "core/data_storage.h"
#include "core/multiprobe_cp.h"
#include "simulation/data_generator.h"
#include <random>
#include <iostream>

using namespace falconn;
using namespace falconn::core;
using namespace ONIAK;
using namespace std;

constexpr int NUM_EXPERIMENTS = 1000;
constexpr int DIMENSION = 1024;
constexpr int PROBES = 1000;

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
  
  CrossPolytopeHash2 cp_hash(DIMENSION, /*k*/ 8, /*l*/ 2, rng(), /*hash width*/ 20);
  ArrayDataStorage<VectorType> data_array(data_vecs);
  typename CrossPolytopeHash2<>::template BatchHash<ArrayDataStorage<VectorType>> batch_hash(cp_hash);
  typename CrossPolytopeHash2<>::HashTransformation hash_tran(cp_hash);
  MultiProbeCP<CrossPolytopeHash2<>> multiprobe(cp_hash, PROBES);
  
  std::vector<HashType> hash1, hash2;
  batch_hash.batch_hash_single_table(data_array, /*l*/ 0, &hash1);
  batch_hash.batch_hash_single_table(data_array, /*l*/ 1, &hash2);

  std::vector<std::vector<HashType>> hashes = {hash1, hash2};
  using MultiprobeVector = std::vector<typename CrossPolytopeHash2<>::MultiprobeType>;
  MultiprobeVector mp_vec;
  for (int exp = 0; exp < NUM_EXPERIMENTS; ++exp) {
    auto data = data_vecs[exp];
    hash_tran.apply(std::make_pair(data, data), &mp_vec, PROBES);
    multiprobe.setup_probing(std::move(mp_vec), 2);
    HashType probe;
    int_fast32_t table;
    multiprobe.get_next_probe(&probe, &table);
    assert(hashes[table][exp] == probe);
    multiprobe.get_next_probe(&probe, &table);
    assert(hashes[table][exp] == probe);
  }
  cout << "Test passed." << endl;
  return 0;
}