#ifndef __ONIAK_SIMULATION_SAMPLER_H__
#define __ONIAK_SIMULATION_SAMPLER_H__

#include "../falconn_global.h"
#include "../fileio.h"
#include "../kronecker/utils.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <bitset>

/* This sampler is faster than ONIAKSampler.
Use this sampler exclusively for studying sample accuracy.
Time measurements are not accurate in the add-on setting.*/

namespace falconn {

constexpr int MaxNumberHashTables = 512;
using FalconnBitset = std::bitset<MaxNumberHashTables>;

class ONIAKSimulationSampler {
 public:
  ONIAKSimulationSampler(std::string mp_recall_path, int num_tables, float bucket_width,
  const std::vector<std::vector<int>>& candidates, int num_data, int max_attempt,
  VectorType distance): 
  bucket_width_(bucket_width), max_attempt_(max_attempt), 
  num_tables_(candidates.size()), num_active_tables_(num_tables), candidates_(num_data),
  tables_(num_tables_), distance_(std::move(distance)),
  rejection_sampler_(0.0, 1.0)  {
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

    std::iota(tables_.begin(), tables_.end(), 0);
    assert(candidates.size() <= MaxNumberHashTables);
    int table = 0;
    for (auto& candi_table : candidates) {
      for (int candi : candi_table) {
        candidates_[candi].set(table);
      }
      ++table;
    }

    std::vector<float> max_distances(num_tables_ + 1, 0.0);
    for (int candi = 0; candi < num_data; ++candi) {
      int active_cnt = candidates_[candi].count();
      max_distances[active_cnt] = std::max(max_distances[active_cnt], distance_(candi));
    }
    float boundary = 0.0;
    for (int cnt = num_tables_; cnt >=0; --cnt) {
      if (max_distances[cnt] > boundary) {
        boundary = max_distances[cnt];
        boundaries_.push_back(boundary);
        double hit_ratio = static_cast<double>(cnt) / static_cast<double>(num_tables_);
        recalls_.push_back(1.0 - std::pow(1.0 - hit_ratio, num_active_tables_));
      }
    }

  }

  // samples candidates among those whose distance <= radius_R
  int sample(const std::vector<int>& neighbors, float radius_R, std::mt19937& rng) const {
    auto active_tables = sample_tables(rng);
    int neighbor_size = neighbors.size();
    assert(neighbor_size >= 1);
    std::uniform_int_distribution<> candi_sampler(0, neighbor_size-1);
    // recall at radius_R
    float boundary_recall = get_recall(radius_R);
    for (int cnt = 0; cnt < max_attempt_; ++cnt) {
      // Uniform sample from all candidates in range
      size_t random_neighbors = candi_sampler(rng);
      if (random_neighbors < 0 || random_neighbors >= neighbors.size()) {
        std::cout << random_neighbors << "\t" << neighbors.size() << std::endl;
      }
      int cid = neighbors[random_neighbors];
      if (cid >= 10000000) {
        std::cout << (neighbor_size-1) << std::endl;
      }
      // Continue if this candidate is not in any active table
      auto active_intersection = candidates_[cid] & active_tables;
      if (active_intersection.none()) continue;
 
      float recall = get_recall(distance_(cid));
      // rejection sampling
      float accept_prob = boundary_recall / recall;
      float accept_sample = rejection_sampler_(rng);
      if (accept_sample <= accept_prob) {
        return cid;
      }
    }
    return -1;  // failure
  }

  // samples candidates among those whose distance <= radius_R
  // This method is deprecated. Cannot be implemented in reasonable time!
  int sample2(const std::vector<int>& neighbors, float radius_R, std::mt19937& rng) const {
    auto active_tables = sample_tables(rng);
    int neighbor_size = neighbors.size();
    std::uniform_int_distribution<> candi_sampler(0, neighbor_size-1);
    // recall at radius_R
    int boundary_bound = std::upper_bound(boundaries_.begin(), boundaries_.end(), radius_R)
        - boundaries_.begin();
    float boundary_recall = recalls_[boundary_bound];

    for (int cnt = 0; cnt < max_attempt_; ++cnt) {
      // Uniform sample from all candidates in range
      int cid = neighbors[candi_sampler(rng)];
      // Continue if this candidate is not in any active table
      auto active_intersection = candidates_[cid] & active_tables;
      if (active_intersection.none()) continue;

      int infer_count = (candidates_[cid] & (~active_tables)).count();
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

  // uniform sample in near neighbors, used for comparison
  int sample3(const std::vector<int>& neighbors, float radius_R, std::mt19937& rng) const {
    int neighbor_size = neighbors.size();
    std::uniform_int_distribution<> candi_sampler(0, neighbor_size-1);
    return neighbors[candi_sampler(rng)];
  }

  int sample_tbl(const std::vector<int>& neighbors, float radius_R, std::mt19937& rng) const {
    auto active_tables = sample_tables(rng);
    int neighbor_size = neighbors.size();
    std::uniform_int_distribution<> candi_sampler(0, neighbor_size-1);
    // recall at radius_R
    float boundary_recall = get_recall(radius_R);

    for (int cnt = 0; cnt < max_attempt_; ++cnt) {
      // Uniform sample from all candidates in range
      int cid = neighbors[candi_sampler(rng)];
      // Continue if this candidate is not in any active table
      auto active_intersection = candidates_[cid] & active_tables;
      if (active_intersection.none()) continue;

      int infer_count = (candidates_[cid] & (~active_tables)).count();
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

  FalconnBitset sample_tables(std::mt19937& rng) const {
    std::vector<int> samples;
    samples.reserve(num_active_tables_);
    std::sample(tables_.begin(), tables_.end(), std::back_inserter(samples),
        num_active_tables_, rng);
    return ONIAK::vector_to_bitset<int, MaxNumberHashTables>(samples);
  }

  double infer_recall(int count) const {
    double recall_one = static_cast<double>(count) / static_cast<double>(num_tables_ - num_active_tables_);
    return 1.0 - std::pow(1.0 - recall_one, num_active_tables_);
  }

  float dist_begin_, dist_step_, dist_end_;
  float bucket_width_;
  int max_attempt_, num_tables_, num_active_tables_;
  std::vector<FalconnBitset> candidates_;
  std::vector<int> tables_;
  std::vector<float> boundaries_, recalls_;
  VectorType mp_recalls_, distance_;
  mutable std::uniform_real_distribution<float> rejection_sampler_;
};

}

#endif