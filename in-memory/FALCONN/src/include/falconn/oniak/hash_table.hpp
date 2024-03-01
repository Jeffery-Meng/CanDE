#ifndef __ONIAK_HASH_TABLE_HPP__
#define __ONIAK_HASH_TABLE_HPP__

#include <bitset>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#include "../kronecker/utils.h"
#include "hash_wrapper.hpp"
#include "../half.hpp"

namespace ONIAK {

// Fast flat hashmap designed specially for CanDE project

struct TableBinPair {
  uint8_t table, bin;
};

struct TableDistancePair {
  uint8_t table;
  half_float::half distance;
}; //__attribute__((packed));

enum class HashInsertionStatus {
  kNewlyInserted, kAlreadyExists, kTableFull
};

template<typename KeyType = int, typename ValueType = TableBinPair,
  typename SignType = uint16_t, size_t BucketSize = 96>
class ONIAKHT {
  template <size_t N>
  struct Bucket;
public:
  //using SVPair = std::pair<SignType, ValueType>;
  // For CanDE we want to minimize memory usage
  // static_assert(sizeof(ValueType) == 1);
  // static_assert(sizeof(SignType) == 1);
  //static_assert(sizeof(SVPair) == 2);
  static constexpr int bucket_size_ = BucketSize / (sizeof(SignType) + sizeof(ValueType));

  // Recommended configuration of 64K buckets, with 32 bins per bucket
  ONIAKHT(std::seed_seq& seeds, int table_size = 131072) :
    table_size_(table_size),
    key_is_sign_(std::is_same_v<KeyType, SignType>),
    table_(table_size_){
    // static_assert(sizeof(TableDistancePair) == 3);
    std::vector<std::uint32_t> seq(2);
    seeds.generate(seq.begin(), seq.end());
    hash_.seed() = seq[0];
    sign_hash_.seed() = seq[1];
    hash_.mask() = static_cast<uint32_t>(table_size - 1);
    sign_hash_.mask() = std::numeric_limits<SignType>::max();
  }

  ValueType& find_or_insert(KeyType key, HashInsertionStatus& status) {
    SignType signature = key_is_sign_? key: sign_hash_(key);
    signature = (signature == 0) ? 1 : signature;
    unsigned hash_val = hash_(key);
    bool is_found;
    int load = -1;
    ValueType& value = table_[hash_val].find(signature, is_found, load);
    if (is_found) {
      status = HashInsertionStatus::kAlreadyExists;
      return value;
    } else if (load == bucket_size_) {
      // all tables are full
      status = HashInsertionStatus::kTableFull;
      return dummy_;
    } else {
      int idx = hash_(key);
      status = HashInsertionStatus::kNewlyInserted;
      return table_[idx].create(signature, load);
    }
  }

  ValueType& find(KeyType key, bool& is_found) {
    SignType signature = key_is_sign_ ? key : sign_hash_(key);
    signature = (signature == 0) ? 1 : signature;
    unsigned hash_val = hash_(key);
    int load = -1;
    return table_[hash_val].find(signature, is_found, load);
  }

  void prefetch(KeyType key) {
    unsigned hash_val = hash_(key);
    __builtin_prefetch(&table_[hash_val], /*write*/0, /*highly local*/2);
  }

  void clear() {
    for (auto& bucket : table_) bucket.clear();
  }

  void resize(size_t sz) {
    assert(IsPowerOfTwo(sz));
    hash_.mask() = static_cast<uint32_t>(sz - 1);
    table_size_ = sz;
    table_.resize(sz);
  }

  size_t size() const { return table_size_; }

  class Iterator {
  public:
    Iterator(int bucket, int load, const std::vector<Bucket<bucket_size_>>& table) :
      table_(table), bucket_(bucket), load_(load) {
      if (bucket_ == 0 && load_ == 0) find_next();
    }
    const ValueType& operator* () const {
      return table_[bucket_].store_[load_];
    }
    Iterator& operator++() {
      ++load_;
      find_next();
      return *this;
    }
    bool operator!=(const Iterator& other) const {
      return !(bucket_ == other.bucket_ && load_ == other.load_);
    }
  private:
    void find_next() {
      while (bucket_ < static_cast<int>(table_.size()) && table_[bucket_].signature_[load_] == 0) {
        ++bucket_;
        load_ = 0;
      }
    }

    const std::vector<Bucket<bucket_size_>>& table_;
    int bucket_, load_;
  };

  Iterator begin() const {
    return { 0, 0, table_ };
  }
  Iterator end() const {
    return { table_size_, 0, table_ };
  }

private:
  template <size_t N>
  struct Bucket {
    Bucket() {
      signature_[0] = 0;
    }
    SignType signature_[N];
    ValueType store_[N];

    ValueType& create(SignType signature, int load) {
      signature_[load] = signature;
      if (load < static_cast<int>(N - 1)) signature_[load + 1] = 0;
      return store_[load];
    }

    bool remove(SignType signature) {
      int i = 0;
      for (; i < static_cast<int>(N); ++i) {
        if (signature_[i] == signature) {
          break;
        }
      }

      if (i == N) {  // signature not found
        return false;
      }
      int j = i+1;
      for (; j < static_cast<int>(N); ++j) {
        if (signature_[j] == 0) {
          break;
        }
      }
    }

    void clear() {
      signature_[0] = 0;
    }

    // if signature exists, valid is true, and load is the offset of that signature in bucket
    // otherwise valid is false, and load is the offset of the first empty slot
    ValueType& find(SignType signature, bool& valid, int& load) {
      for (load = 0; load < static_cast<int>(N); ++load) {
        if (signature_[load] == signature) {
          valid = true;
          return store_[load];
        }
        if (signature_[load] == 0) break;
      }
      valid = false;
      return store_[0];;
    }

    friend class Iterator;
  } __attribute__((aligned(8)));;

  ValueType dummy_;
  WYHash hash_, sign_hash_;
  int table_size_;
  // use key as signature instead of hashed key
  bool key_is_sign_;
  // number of elements per bucket
  std::vector<Bucket<bucket_size_>> table_;

  // static_assert(sizeof(Bucket<bucket_size_>) == 96);
};



}
#endif 