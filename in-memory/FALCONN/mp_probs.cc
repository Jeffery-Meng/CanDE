#include "falconn/config.h"
#include "falconn/fileio.h"
#include "falconn/kronecker/utils.h"
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "boost/unordered_set.hpp"

/* Compute the retrival probablity of each candidate. */

using namespace falconn;
using namespace std;

constexpr double kUB = 2;
constexpr double kStep = 0.0001;

int id_translate(float val) {
  int64_t result = (val + 5) * 1000000;
  if (result < 0) result = 0;
  else if (result >= 10000000) result = 9999999;
  return result;
}

double bucket_prob(const VectorType& gaussian_cdf, const VectorType& lower,
    const VectorType& upper) {
    double result = 1.0;
    int sz = lower.size();
    for (int i = 0; i < sz; ++i) {
      int lower_id = id_translate(lower(i));
      int upper_id = id_translate(upper(i));
      result *= gaussian_cdf[upper_id] - gaussian_cdf[lower_id];
    }
    return result;
}

int main(int argc, char * argv[]) {
  if (argc < 2) {
    cout << "Usage: ./candidate_probs [config file]";
  }

   
  LSHConstructionParameters conf = read_config(argv[1]);
  int hash_funcs_per_table = conf.hash_table_params[0].k;
  int hash_table_width = conf.hash_table_width;
  int probes_per_table = conf.probes_per_table;
  int num_distances = 0;
  for (double distance = kStep; distance <= kUB; distance += kStep) {
    ++num_distances;
  }
  // Probablity of spontaneous collision due to low hash table width.
  double spontaneous_coll_prob = std::pow(2, -hash_table_width);
  auto mp_sequence_raw = read_data<DenseVector<int>>(conf.probing_sequence_file);
  auto gaussian_cdf = read_data<VectorType>(conf.eigen_filename)[0];

  std::vector<VectorType> mp_sequence;
  mp_sequence.push_back(VectorType::Zero(hash_funcs_per_table));
  for (auto line : mp_sequence_raw) {
    VectorType bucket = VectorType::Zero(hash_funcs_per_table);
    for (auto val : line) {
      if (val <= hash_funcs_per_table) {
        bucket[val - 1] = 1.0;
      } else {  // on the other (far) side
        bucket[val - hash_funcs_per_table - 1] = -1.0;
      }
      
    }
    mp_sequence.push_back(std::move(bucket));
  }

  std::mt19937 rng(conf.seed);
  std::uniform_real_distribution<float> unif_real(0, 0.5);
  std::vector<double> recall_ps(num_distances, 0.0);
  assert(conf.num_experiments > 0);
  for (int exp = 0; exp < conf.num_experiments; ++exp) {
    VectorType random_hash(hash_funcs_per_table);
    if (exp % 10 == 0) {
      std::cout << exp << std::endl;
    }
    for (auto& val : random_hash) {
      val = unif_real(rng);
    }
    std::sort(random_hash.begin(), random_hash.end());
    for (int bucket = 0; bucket < probes_per_table; ++bucket) {
      auto& bucket_vec = mp_sequence[bucket];
      VectorType lower_vec = -bucket_vec - random_hash;
      VectorType upper_vec = lower_vec + VectorType::Ones(hash_funcs_per_table);
      int distance_idx = 0;
      for (double distance = kStep; distance <= kUB; distance += kStep) {
          // normalize to standard gaussian
          VectorType lower_vec_norm = lower_vec / distance;
          VectorType upper_vec_norm = upper_vec / distance;
          recall_ps[distance_idx++] += bucket_prob(gaussian_cdf, lower_vec_norm, upper_vec_norm)
              + spontaneous_coll_prob;
      }
    }
  }
  for (auto& val : recall_ps) {
    val /= conf.num_experiments;
  }

  std::ofstream fout(conf.mp_prob_filename);
  write_data(fout, std::vector<float>({kStep, kUB, kStep}));
  write_data(fout, recall_ps);
  std::cout << "Multi-probe probabilities of l = " << hash_funcs_per_table <<
      " written to " << conf.mp_prob_filename << std::endl;
  return 0;
}