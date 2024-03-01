#ifndef __ONIAK_SAMPLER_H__
#define __ONIAK_SAMPLER_H__

#include "../falconn_global.h"
#include "../fileio.h"
#include <cmath>
#include <fstream>
#include <random>

namespace falconn {

class ONIAKSampler {
 public:
  ONIAKSampler(std::string mp_recall_path, int num_tables, float bucket_width, int max_attempt): 
  bucket_width_(bucket_width), max_attempt_(max_attempt) {
    auto recall_file = read_data<VectorType>(mp_recall_path);
    assert(recall_file.size() >= 2);
    auto& dist_configs = recall_file[0];
    assert(dist_configs.size() >=3);
    dist_begin_ = dist_configs[0];
    dist_step_ = dist_configs[2];
    dist_end_ = dist_configs[1];
    mp_recalls_.resize(recall_file[1].size());
    for (int idx = 0; idx < recall_file[1].size(); ++idx) {
      mp_recalls_(idx) = recall_with_tables(num_tables, recall_file[1](idx));
    }
  }

  // samples candidates among those whose distance <= radius_R
  int sample(const std::vector<int>& candidates, const std::vector<float>& distances,
      float radius_R, std::mt19937& rng) const {
    assert(candidates.size() > 0);
    std::uniform_int_distribution<> candi_sampler(0, candidates.size()-1);
    std::uniform_real_distribution<float> rejection_sampler(0.0, 1.0);
    // recall at radius_R
    float boundary_recall = get_recall(radius_R);
    for (int cnt = 0; cnt < max_attempt_; ++cnt) {
      int cid = candi_sampler(rng);
      if (distances[cid] > radius_R) continue;
      float recall = get_recall(distances[cid]);
      // rejection sampling
      float accept_prob = boundary_recall / recall;
      float accept_sample = rejection_sampler(rng);
      if (accept_sample <= accept_prob) {
        return candidates[cid];
      }
    }
    return -1;  // failure
  }

// sampling when taking the time of distance calculations into account.
  int sample_with_distance(const std::vector<int>& candidates, 
      const std::function<float(int)>& distance_func,
      float radius_R, std::mt19937& rng) const {
    assert(candidates.size() > 0);
    std::uniform_int_distribution<> candi_sampler(0, candidates.size()-1);
    std::uniform_real_distribution<float> rejection_sampler(0.0, 1.0);
    // recall at radius_R
    float boundary_recall = get_recall(radius_R);
    for (int cnt = 0; cnt < max_attempt_; ++cnt) {
      int cid = candi_sampler(rng);
      if (distance_cache_[cid] < 0) {
        distance_cache_[cid] = distance_func(cid);
      }
      if (distance_cache_[cid] > radius_R) continue;
      float recall = get_recall(distance_cache_[cid]);
      // rejection sampling
      float accept_prob = boundary_recall / recall;
      float accept_sample = rejection_sampler(rng);
      if (accept_sample <= accept_prob) {
        return candidates[cid];
      }
    }
    return -1;  // failure
  }

  void reset_cache(int num_data) {
    distance_cache_.assign(num_data, -1.0);
  }

 private:
  int id_translate(float distance) const {
    distance /= bucket_width_;
    int id = (distance - dist_begin_) / dist_step_;
    if (id < 0) {
      id = 0;
    } else if (id >= mp_recalls_.size()) {
      id = mp_recalls_.size() - 1;
    }
    return id;
  }

  float get_recall(float distance) const { return mp_recalls_[id_translate(distance)]; }

  static float recall_with_tables(int l, double recall) {
    return 1.0 - std::pow(1.0 - recall, l);
  }

  float dist_begin_, dist_step_, dist_end_;
  float bucket_width_;
  int max_attempt_;
  VectorType mp_recalls_;
  mutable std::vector<float> distance_cache_;
};

}

#endif