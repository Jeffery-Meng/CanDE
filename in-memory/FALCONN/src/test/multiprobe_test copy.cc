#include "kronecker/kronecker_lsh.h"
#include "simulation/data_generator.h"
#include <random>
#include <iostream>

using namespace falconn;
using namespace ONIAK;
using namespace std;

constexpr int NUM_EXPERIMENTS = 1000;
constexpr int DIMENSION = 1024;
constexpr int PROBES = 1000;

int main(int argc, char * argv[]) {
  std::random_device rd;
  uint32_t seed_alt;
  if (argc > 1) seed_alt = atoi(argv[1]);
  for (int exp = 0; exp < NUM_EXPERIMENTS; ++exp) {
  uint32_t seed  = rd();
  if (argc > 1) seed = seed_alt;
  std::mt19937 rng(3829432);
    DRowVector v1 = UniformRandomVector(DIMENSION, rng),
               v2 = UniformRandomVector(DIMENSION, rng);
    
    KroneckerLSH lsh(DIMENSION, rng);

    auto list1 = lsh.multiprobe(v1, v2, PROBES);
    auto list2 = lsh.multiprobe_slow(v1, v2, PROBES);
    cout << (list1 == list2) << endl;
    if (list1 != list2) {
      cout << "error\t" << exp << "\t" << seed << endl;
      exit(1);
    }
  }

  return 0;
}