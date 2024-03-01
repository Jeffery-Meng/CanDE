#ifndef __FAIR_NN_SAMPLER_H__
#define __FAIR_NN_SAMPLER_H__

#include "../falconn_global.h"
#include "../fileio.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <random>

namespace falconn {

class FairNNSampler {
 public:
  explicit FairNNSampler(int max_attempt): max_attempt_(max_attempt),
  rejection_sampler_(0.0, 1.0) {}

  // samples uniformly from the union of candidates
  int sample(const std::vector<std::vector<int>>& candidate_vecs,
      const std::vector<HashSet<int>>& candidates,
      const std::function<float(int)>& distance_func,
      float radius_R, std::mt19937& rng, bool use_cached_distance = false) const {
    int num_tables = candidates.size();
    std::vector<int> sum_sizes(num_tables + 1, 0);
    int sum = 0, idx = 0;
    std::generate(sum_sizes.begin() + 1, sum_sizes.end(), [&]() {
      sum += candidates[idx++].size(); return sum;
    });
    std::uniform_int_distribution<> candi_sampler(0, sum-1);
    std::uniform_real_distribution<float> rejection_sampler(0.0, 1.0);
    
    for (int cnt = 0; cnt < max_attempt_; ++cnt) {
      int cid = candi_sampler(rng);
      // sum_sizes[selected_table + 1] is the first element that is greater than cid.
      int selected_table = std::upper_bound(sum_sizes.begin(), sum_sizes.end(), cid) - sum_sizes.begin() - 1;
      int bias_in_table = cid - sum_sizes[selected_table];
      int candi = *(candidate_vecs[selected_table].begin() + bias_in_table);

      if (!use_cached_distance || distance_cache_[candi] < 0) {
        distance_cache_[candi] = distance_func(candi);
      }
      if (distance_cache_[candi] > radius_R) continue;

      float accept_prob = 1.0 / approximate_degree(candidates, candi, rng);
      float accept_sample = rejection_sampler(rng);
      if (accept_sample <= accept_prob) {
        return candi;
      }
    }
    return -1;  // failure
  }

  void reset_cache(int num_data) {
    distance_cache_.assign(num_data, -1.0);
  }

  // does not check for radius, used for simulating accuracy
  int sample_fast(const std::vector<std::vector<int>>& candidate_vecs,
      const std::vector<HashSet<int>>& candidates,
      std::mt19937& rng) const {
    
    if (sum_ == 0) {
    // No neighbors found. Failure.
      return -1;
    }
    std::uniform_int_distribution<> candi_sampler(0, sum_-1);
    for (int cnt = 0; cnt < max_attempt_; ++cnt) {
      int cid = candi_sampler(rng);
      // sum_sizes[selected_table + 1] is the first element that is greater than cid.
      int selected_table = std::upper_bound(sum_sizes_.begin(), sum_sizes_.end(), cid) - sum_sizes_.begin() - 1;
      int bias_in_table = cid - sum_sizes_[selected_table];
      int candi = *(candidate_vecs[selected_table].begin() + bias_in_table);

      float accept_prob = 1.0 / approximate_degree(candidates, candi, rng);
      float accept_sample = rejection_sampler_(rng);
      if (accept_sample <= accept_prob) {
        return candi;
      }
    }
    return -1;  // failure
  }

  // resets state used for 
  void reset_state(const std::vector<HashSet<int>>& candidates) {
    int num_tables = candidates.size();
    sum_sizes_.assign(num_tables + 1, 0);
    int idx = 0;
    sum_ = 0;
    std::generate(sum_sizes_.begin() + 1, sum_sizes_.end(), [&]() {
      sum_ += candidates[idx++].size(); return sum_;
    });
  }

 private:
  float approximate_degree(const std::vector<HashSet<int>>& candidates, int candi,
      std::mt19937& rng) const {
    int num_tables = candidates.size();
    std::uniform_int_distribution<> table_sampler(0, num_tables - 1);
    int cnt = 1;
    while (true) {
      int table = table_sampler(rng);
      if (candidates[table].find(candi) != candidates[table].end()) {
        return static_cast<float> (num_tables) /  cnt;
      }
      ++cnt;
    }
  }


  int max_attempt_, sum_;
  std::vector<int> sum_sizes_;
  mutable std::vector<float> distance_cache_;
  mutable std::uniform_real_distribution<float> rejection_sampler_;
};

}

#endif