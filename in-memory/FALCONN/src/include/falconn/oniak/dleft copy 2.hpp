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

struct TableBinPair{
  uint8_t table, bin;
};

template<typename KeyType = int, typename ValueType = TableBinPair, typename SignType = uint16_t>
class ONIAKDLeft{
 public:
  //using SVPair = std::pair<SignType, ValueType>;
  // For CanDE we want to minimize memory usage
  // static_assert(sizeof(ValueType) == 1);
  // static_assert(sizeof(SignType) == 1);
  //static_assert(sizeof(SVPair) == 2);

  // Recommended configuration of 64K buckets, with 32 bins per bucket
  ONIAKDLeft(std::seed_seq& seeds, int table_size = 131072, int n_tables = 4, int bucket_size = 8):
  total_size_(n_tables * table_size), n_tables_(n_tables), table_size_(table_size),
  bucket_size_(bucket_size), table_(total_size_)
  { 
    std::cout << sizeof(Bucket<19>) << std::endl;
    std::vector<std::uint32_t> seq(2);
    seeds.generate(seq.begin(), seq.end());
    hash_.seed() = seq[0];
    sign_hash_.seed() = seq[2];
    hash_.mask() = static_cast<uint32_t>(table_size - 1);
    sign_hash_.mask() = std::numeric_limits<SignType>::max();
  }

  ValueType& find_or_insert(KeyType key, int& status) {
    SignType signature = sign_hash_(key);
    signature = (signature == 0)? 1: signature;
    int min_table = -1, min_load = bucket_size_;
    for (int table = 0; table < n_tables_; ++ table) {
      unsigned hash_val = hash_(key, table);
      bool is_found;
      int load = -1;
      ValueType& value = table_[table * table_size_ + hash_val].find(signature, is_found, load);
      if (is_found) {
        status = 0;
        return value;
      }
      assert(load >= 0);
      if (load < min_load) {
        min_load = load;
        min_table = table;
      }
    }
    if (min_table == -1) {
      // all tables are full
      status = -1;
      return dummy_;
    } else {
      int idx = min_table * table_size_ + hash_(key, min_table);
      status = 1;
      return table_[idx].create(signature);
    }
  }

  void clear() {
    for (auto& bucket : table_) bucket.clear();
  }

 private:
  template <size_t N>
  struct Bucket{
    Bucket(): load_(0) {}
    SignType signature_[N];
    ValueType store_[N];
    int32_t load_;

    void insert(SignType signature, ValueType value) {
      store_[load_] = value;
      signature_[load_] = signature_;
      ++load_;
    }

    ValueType& create(SignType signature) {
      signature_[load_] = signature;
      return store_[load_++];
    }

    void clear() {
      load_ = 0;
    }

    ValueType& find(SignType signature, bool& valid, int& load) {
      load = load_;
      for (int i = 0; i < load_; ++i) {
        if (signature_[i] == signature) {
          valid = true;
          return store_[i];
        }
      }
      valid = false;
      return store_[0];;
    }
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

  ValueType dummy_;
  WYHash hash_, sign_hash_;
  const int total_size_, n_tables_, table_size_, bucket_size_;
  std::vector<Bucket<19>> table_;
};

}
#endif 