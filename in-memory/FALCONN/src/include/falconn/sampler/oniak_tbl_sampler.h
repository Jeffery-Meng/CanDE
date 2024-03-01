#ifndef __ONIAK_TBL_SAMPLER_H__
#define __ONIAK_TBL_SAMPLER_H__

#include "../falconn_global.h"
#include "../fileio.h"
#include <cmath>
#include <fstream>
#include <random>
#include "oniak_simulation_sampler.h"

/* JFM thinks table-based sampler cannot be implemented in the FNN setting.
Please ignore this file.*/

namespace falconn {

class ONIAKTableSampler {
 public:
  ONIAKTableSampler(std::string mp_recall_path, float bucket_width, 
      int max_attempt, int num_tables, int num_active_tables): 
  bucket_width_(bucket_width), max_attempt_(max_attempt),
  num_tables_(num_tables), num_active_tables_(num_active_tables), rejection_sampler_(0.0, 1.0) 
   {
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
      float radius_R, std::mt19937& rng, const std::vector<FalconnBitset>& candidate_sets, 
      FalconnBitset active_tables) const {
    assert(candidates.size() > 0);
    std::uniform_int_distribution<> candi_sampler(0, candidates.size()-1);
    std::uniform_real_distribution<float> rejection_sampler(0.0, 1.0);
    // recall at radius_R
    float boundary_recall = get_recall(radius_R);
    for (int cnt = 0; cnt < max_attempt_; ++cnt) {
      // Uniform sample from all candidates in range
      int cid = candidates[candi_sampler(rng)];
      if (distances[cid] > radius_R) continue;
      // Continue if this candidate is not in any active table

      int infer_count = (candidate_sets[cid] & (~active_tables)).count();
      float recall = infer_recall(infer_count);
      // rejection sampling
      float accept_prob = boundary_recall / recall;
      float accept_sample = rejection_sampler_(rng);
      if (accept_sample <= accept_prob) {
        return cid;
      }
    }
    return -1;  // failure
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

  double infer_recall(int count) const {
    double recall_one = static_cast<double>(count) / static_cast<double>(num_tables_ - num_active_tables_);
    return 1.0 - std::pow(1.0 - recall_one, num_active_tables_);
  }

  float dist_begin_, dist_step_, dist_end_, bucket_width_;
  VectorType mp_recalls_;
  int max_attempt_, num_tables_, num_active_tables_;
  mutable std::uniform_real_distribution<float> rejection_sampler_;
};

}

#endif