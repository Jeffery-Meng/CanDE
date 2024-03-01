#ifndef __ONIAK_DLEFT_HPP__
#define __ONIAK_DLEFT_HPP__

#include <bitset>
#include <limits>
#include <utility>
#include <vector>
#include "hash_wrapper.hpp"

namespace ONIAK {

// representing a bitset of 16 bits using an integer
class BitSet16 {
public:
// constexpr constructor needed if we want to define constexpr BitSet16
  constexpr BitSet16() : word_(0u) {}
  void set(unsigned i) {
    word_ |= 1u << i;
  }
  bool test(unsigned i) const {
    return word_ & (1u << i);
  }
  int count() const {
    return __builtin_popcount(static_cast<unsigned>(word_));
  }
  class Iterator {
  public:
    Iterator(uint16_t word) : wd_(word) {}
    int operator*() const {
      return __builtin_ctz(wd_);
    }
    const Iterator& operator++() {
      wd_ ^= 1u << **this;
      return *this;
    }
    bool operator==(const Iterator& other) const {
      return wd_ == other.wd_;
    } 
  private:
    unsigned wd_;
  };

  Iterator begin() const {
    return {word_};
  }
  Iterator end() const {
    return {0u};
  }
private:
  uint16_t word_;
};

// DLeft hashmap designed specially for CanDE project


template<typename KeyType = int, typename ValueType = uint8_t, typename SignType = uint8_t>
class ONIAKDLeft{
 public:
  using SVPair = std::pair<SignType, ValueType>;
  // For CanDE we want to minimize memory usage
  // static_assert(sizeof(ValueType) == 1);
  // static_assert(sizeof(SignType) == 1);
  static_assert(sizeof(SVPair) == 2);
  static constexpr SVPair kEmptySVPair = std::make_pair(SignType(0u), ValueType());

  // Recommended configuration of 64K buckets, with 32 bins per bucket
  ONIAKDLeft(std::seed_seq& seeds, int table_size = 131072, int n_tables = 4, int bucket_size = 8):
  total_size_(n_tables * bucket_size * table_size), n_tables_(n_tables), table_size_(table_size),
  bucket_size_(bucket_size), table_(total_size_, Bucket(bucket_size_))
  {
    std::vector<std::uint32_t> seq(2);
    seeds.generate(seq.begin(), seq.end());
    hash_.seed() = seq[0];
    sign_hash_.seed() = seq[2];
    hash_.mask() = static_cast<uint32_t>(table_size - 1);
    sign_hash_.mask() = std::numeric_limits<SignType>::max();
  }

  bool insert(KeyType key, ValueType value) {
    std::vector<unsigned> hashes(n_tables_);
    for (int table = 0; table < n_tables_; ++ table) {
      hashes[table] = hash_(key, table);
    }
    int table = table_to_insert(hashes);
    if (table == -1) {
      // all tables are full
      return false;
    } else {
      int idx = table * table_size_ + hashes[table];
      int load = load_[idx];
      ++load_[idx];
      idx = idx * bucket_size_ + load;
      SignType signature = sign_hash_(key);
      table_[idx] = std::make_pair(signature, table);
      return true;
    }
  }

  std::vector<ValueType> find_and_delete(KeyType key) {
    std::vector<ValueType> result;
    SignType signature = sign_hash_(key);
    for (int table = 0; table < n_tables_; ++ table) {
      int hash = hash_(key, table);
      int bucket_id = table * table_size_ + hash;
      int bucket_start = bucket_id * bucket_size_;
      for (int idx = bucket_start; idx < bucket_start + load_[bucket_id]; ) {
        if (table_[idx].first == signature) { // hit
          result.push_back(table_[idx].second);
          --load_[bucket_id];
          // move the last item to current slot
          table_[idx] = table_[bucket_start + load_[bucket_id]];
        } else { // collision
          ++idx;
        }
      }
    }
    return result;
  }

 private:
  struct Bucket{
    Bucket(int size): load_(0), store_(size, kEmptySVPair)  {}
    std::vector<SVPair> store_;
    int_fast8_t load_;
  };

  int table_to_insert(const std::vector<unsigned>& hashes) {
    int min_load = bucket_size_;
    int result = -1;
    for (int table = 0; table < n_tables_; ++table) {
      int load =  table_[table * table_size_ + hashes[table]].load_;
      if (load < min_load) {
        min_load = load;
        result = table;
      }
    } 
    return result;
  }

  WYHash hash_, sign_hash_;
  const int total_size_, n_tables_, table_size_, bucket_size_;
  std::vector<Bucket> table_;
};

}
#endif 