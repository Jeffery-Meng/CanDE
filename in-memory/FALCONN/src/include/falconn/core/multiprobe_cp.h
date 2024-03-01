#ifndef __MULTIPROBE_CP_H__
#define __MULTIPROBE_CP_H__

#include <vector>
#include "Eigen/Dense"
#include "wyhash32.h"
#include <fstream>

#include "heap.h"
#include "multiprobe.h"
#include "lsh_function_helpers.h"

namespace falconn {
namespace core {

// heaped multiprobe for cross-polytope
template <typename HashFunction>
class MultiProbeCP final: public MultiProbeBase<HashFunction>{
   public:
     typedef typename HashFunction::HashType HashType;
     typedef typename HashFunction::HashTransformation HashTran;
     typedef typename HashFunction::QueryType QueryType;
     typedef typename HashFunction::MultiprobeType MultiprobeType;

    MultiProbeCP(const HashFunction& parent, unsigned num_probes)
        : 
          k_(parent.get_k()), 
          l_(parent.get_l()), hash_width_(parent.hash_table_width()),
          num_probes_(num_probes),
          cur_probe_counter_(0),
          hash_seed_(parent.get_seed()), hash_tran_(parent) {
    }

//  set up the heaps for hash_vector (personalized probing sequence for hash_vector)
    void setup_probing(MultiprobeType mp_vector,
                       int_fast64_t num_probes) override {
      num_probes_ = num_probes;
      cur_probe_counter_ = -1;

      // insert l_ epicenter buckets into heap.
      heap_.reset();
      for (int ll = 0; ll < l_; ++ll) {
        heap_.insert(/*score=*/ 0, ProbeCandidate(/*table=*/ ll, std::vector<int>(k_, 0)));
      }
      // save multiprobe lists
      mp_vector_ = std::move(mp_vector);
    }

  // return probe (the bucket of this probe) and table (by pointer)
    bool get_next_probe(HashType* cur_probe, int_fast32_t* cur_table) override {
      cur_probe_counter_ += 1;
      if (num_probes_ >= 0 && cur_probe_counter_ >= num_probes_ && heap_.empty()) {
        // printf("out of probes\n");
        return false;
      }

      CoordinateType score;
      ProbeCandidate candidate;
      heap_.extract_min(&score, &candidate);
      *cur_table = candidate.table_;
      *cur_probe = get_bucket(candidate);

      bool breakout = false;
      // Starting from (0, ... 0), only increment hash functions
      // before (including) the first non-zero value.
      // For example (0, 0, 0) -> (0, 0, 1) -> (1, 0, 1) -> (2, 0, 1)
      // In this way, we avoid duplicate buckets in the heap.
      for (int kk = 0; kk < k_; ++kk) {
        if (candidate.orders_[kk] != 0) breakout = true;
        ProbeCandidate new_candidate = candidate;
        CoordinateType new_score = score;
        ++new_candidate.orders_[kk];
        if (new_candidate.orders_[kk] < num_probes_) {
          new_score += mp_vector_[*cur_table * k_ + kk][new_candidate.orders_[kk]].score -
                      mp_vector_[*cur_table * k_ + kk][candidate.orders_[kk]].score;
          heap_.insert(new_score, new_candidate);
        }
        if (breakout) break;
      }
      return true;
    }

       // last_index is used for generating the next perturbation vector
   // it is the largest (latest) dimension of the perturbation vector
   // wipe_mask is 1->unchanged 0-> changed, used to wipe out 
   // pert_mask is 0->unchanged [new hash bucket]-> changed
    class ProbeCandidate {
     public:
      explicit ProbeCandidate(int_fast32_t table = 0)
          : table_(table) {}
      ProbeCandidate(int_fast32_t table, std::vector<int> orders)
          : table_(table), orders_(std::move(orders)) {}
      int_fast32_t table_;
      std::vector<int> orders_;
    };

    HashType get_bucket(const ProbeCandidate& candidate) const {
      int table = candidate.table_;
      std::vector<int> indices(k_, 0);
      for (int kk = 0; kk < k_; ++kk) {
        indices[kk] = mp_vector_[table * k_ + kk][candidate.orders_[kk]].index;
      }
      HashType hash_value = wyhash32(indices.data(), this->k_ * sizeof(int), hash_seed_);
      return hash_value & ( (1<<hash_width_) - 1);
    } 

    // Returns the number of probes in each hash table
    int_fast64_t num_probes() const { return num_probes_ / l_;}

    int_fast32_t k_;
    int_fast32_t l_, hash_width_;
    int_fast64_t num_probes_;
    int_fast64_t cur_probe_counter_;
    uint_fast32_t hash_seed_;
    SimpleHeap<CoordinateType, ProbeCandidate> heap_;
    // h(data), center of probing
    std::vector<int_fast32_t> hash_vector_;
    HashTran hash_tran_;
    MultiprobeType mp_vector_;
};

// a slow, brute-force class used for debugging.
template <typename HashFunction>
class MultiProbeCPDebug : public MultiProbeBase<HashFunction>{
   public:
     typedef typename HashFunction::HashType HashType;
     typedef typename HashFunction::HashTransformation HashTran;
     typedef typename HashFunction::QueryType QueryType;
     typedef typename HashFunction::MultiprobeType MultiprobeType;

    MultiProbeCPDebug(const HashFunction& parent, unsigned num_probes)
        : 
          k_(parent.get_k()), 
          l_(parent.get_l()), hash_width_(parent.hash_table_width()),
          num_probes_(num_probes),
          cur_probe_counter_(0),
          hash_seed_(parent.get_seed()), hash_tran_(parent) {
    }

//  set up the heaps for hash_vector (personalized probing sequence for hash_vector)
    void setup_probing(MultiprobeType mp_vector,
                       int_fast64_t num_probes) override {
      num_probes_ = num_probes;
      cur_probe_counter_ = -1;
      mp_vector_ = std::move(mp_vector);
      
      int capacity = l_;
      for (int kk = 0; kk < k_; ++kk) {
        capacity *= mp_vector_[kk].size();
      }
      buckets_.clear();
      buckets_.reserve(capacity);
      std::vector<int> indices;
      for (int table = 0; table < l_; ++table) {
        recursive_add(indices, 0, table);
      }
      
      std::sort(buckets_.begin(), buckets_.end());
    }

    void recursive_add(std::vector<int>& indices, CoordinateType score, int_fast32_t table) {
      if (indices.size() == k_) {
        HashType hash_value = wyhash32(indices.data(), this->k_ * sizeof(int), hash_seed_);
        hash_value &= (1<<hash_width_)-1;
        buckets_.push_back(ProbeHash{hash_value, score, table});
      } else {
        // position of current hash function
        size_t pos = table * k_ + indices.size();
        for (const auto& candidate : mp_vector_[pos]) {
          indices.push_back(candidate.index);
          recursive_add(indices, score + candidate.score, table);
        }
      }
      if (indices.size() > 0){
        indices.pop_back();
      }
    }

  // return probe (the bucket of this probe) and table (by pointer)
    bool get_next_probe(HashType* cur_probe, int_fast32_t* cur_table) override {
      cur_probe_counter_ += 1;
      if (num_probes_ >= 0 && cur_probe_counter_ >= num_probes_) {
        // printf("out of probes\n");
        return false;
      }

      const ProbeHash& probe = buckets_[cur_probe_counter_];
      *cur_table = probe.table;
      *cur_probe = probe.hash;
      return true;
    }

    struct ProbeHash {
      HashType hash;
      CoordinateType score;
      int_fast32_t table;
      bool operator<(const ProbeHash& hash) {return score < hash.score; }
    };

    int_fast64_t num_probes() const { return num_probes_;}

    int_fast32_t k_;
    int_fast32_t l_, hash_width_;
    int_fast64_t num_probes_;
    int_fast64_t cur_probe_counter_;
    uint_fast32_t hash_seed_;
    // h(data), center of probing
    std::vector<int_fast32_t> hash_vector_;
    HashTran hash_tran_;
    MultiprobeType mp_vector_;
    std::vector<ProbeHash> buckets_;
};
}}

#endif