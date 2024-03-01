#ifndef __FILE_IO_H__
#define __FILE_IO_H__

#include "lsh_nn_table.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <utility>

namespace falconn {

template<typename T>
inline constexpr bool always_false_v = false;


template <typename PointType>
auto read_data(std::string filename) -> std::vector<PointType>{
  static_assert(always_false_v<PointType>, "Not supported point type!");
}

template <>
auto read_data<VectorType>(std::string filename) -> std::vector<VectorType> {
  static_assert(std::is_same_v<float&, decltype(std::declval<VectorType>()[0])>);
  std::ifstream fin(filename);
  assert(fin);
  int sz;
  fin.read(reinterpret_cast<char*> (&sz), 4);
  std::vector<VectorType> result;
  while (fin) {
    VectorType data(sz);
    fin.read(reinterpret_cast<char*> (data.data()), sizeof(float) * sz);
    fin.read(reinterpret_cast<char*> (&sz), 4);
    result.push_back(std::move(data));
  }
  return result;
}

template <>
auto read_data<std::vector<float>>(std::string filename) -> std::vector<std::vector<float>> {
  std::ifstream fin(filename);
  if (!fin) {
    std::cerr << filename << " not opened!" << std::endl;
    throw std::filesystem::filesystem_error("File not opened", filename, std::make_error_code(std::errc::io_error));
  }
  int sz;
  fin.read(reinterpret_cast<char*> (&sz), 4);
  std::vector<std::vector<float>> result;
  while (fin) {
    std::vector<float> data(sz);
    fin.read(reinterpret_cast<char*> (data.data()), sizeof(float) * sz);
    fin.read(reinterpret_cast<char*> (&sz), 4);
    result.push_back(std::move(data));
  }
  return result;
}


template <>
auto read_data<std::vector<double>>(std::string filename) -> std::vector<std::vector<double>> {
  assert(std::filesystem::path(filename).extension() == ".dvecs");
  std::ifstream fin(filename);
  if (!fin) {
    std::cerr << filename << " not opened!" << std::endl;
    throw std::filesystem::filesystem_error("File not opened", filename, std::make_error_code(std::errc::io_error));
  }
  int sz;
  fin.read(reinterpret_cast<char*> (&sz), 4);
  std::vector<std::vector<double>> result;
  while (fin) {
    std::vector<double> data(sz);
    fin.read(reinterpret_cast<char*> (data.data()), sizeof(double) * sz);
    fin.read(reinterpret_cast<char*> (&sz), 4);
    result.push_back(std::move(data));
  }
  return result;
}

template <>
auto read_data<std::vector<int>>(std::string filename) -> std::vector<std::vector<int>> {
  std::ifstream fin(filename);
  if (!fin) {
    std::cerr << filename << " not opened!" << std::endl;
    throw std::filesystem::filesystem_error("File not opened", filename, std::make_error_code(std::errc::io_error));
  }
  int sz;
  fin.read(reinterpret_cast<char*> (&sz), 4);
  std::vector<std::vector<int>> result;
  while (fin) {
    std::vector<int> data(sz);
    fin.read(reinterpret_cast<char*> (data.data()), sizeof(int) * sz);
    fin.read(reinterpret_cast<char*> (&sz), 4);
    result.push_back(std::move(data));
  }
  return result;
}

template <>
auto read_data<DenseVector<int>>(std::string filename) -> std::vector<DenseVector<int>> {
  std::ifstream fin(filename);
  assert(fin);
  int sz;
  fin.read(reinterpret_cast<char*> (&sz), 4);
  std::vector<DenseVector<int>> result;
  while (fin) {
    DenseVector<int> data(sz);
    fin.read(reinterpret_cast<char*> (&data(0)), sizeof(int) * sz);
    fin.read(reinterpret_cast<char*> (&sz), 4);
    result.push_back(std::move(data));
  }
  return result;
}

template <>
auto read_data<VectorPairType>(std::string filename) -> std::vector<VectorPairType> {
  static_assert(std::is_same_v<float&, decltype(std::declval<VectorPairType>().first[0])>);
  std::ifstream fin(filename);
  assert(fin);
  int sz;
  fin.read(reinterpret_cast<char*> (&sz), 4);
  const int cols = sz;
  std::vector<VectorPairType> result;
  while (fin) {
    VectorPairType data = std::make_pair(VectorType(sz), VectorType(sz));
    fin.read(reinterpret_cast<char*> (&data.first(0)), sizeof(float) * sz);
    fin.read(reinterpret_cast<char*> (&sz), 4);
    assert(sz == cols);
    fin.read(reinterpret_cast<char*> (&data.second(0)), sizeof(float) * sz);
    fin.read(reinterpret_cast<char*> (&sz), 4);
    assert(sz == cols);
    result.push_back(std::move(data));
  }
  return result;
}

template <typename PointType>
class QueryIterator {
public:
  QueryIterator() {
  static_assert(always_false_v<PointType>, "Not supported point type!");
}
};

// query iterator for vector pairs

template <>
class QueryIterator<VectorPairType> {
public:
  QueryIterator(std::string raw_query_file, std::string row_file) :
  queries_raw_(read_data<VectorType>(raw_query_file)),
  row_ids_(read_data<DenseVector<int>>(row_file)),
  qn_(row_ids_.size()) {}

  QueryIterator() = default;

  QueryIterator(std::string raw_query_file, int qn) :
  queries_raw_(read_data<VectorType>(raw_query_file)),
  qn_(qn) {
    row_ids_.resize(qn_);
    int i = 0;
    for (auto& row : row_ids_) {
      row.resize(2);
      row << 2 * i, 2 * i + 1;
      ++i;
    }
  }
  
  class Iter {
    int id_;
    const QueryIterator& parent_;
   public:
    explicit Iter(int id, const QueryIterator& parent): id_(id), parent_(parent) {}
    Iter& operator++() {++id_; return *this; }
    bool operator==(const Iter& other) {return id_ == other.id_; }

    VectorPairRefType operator*() {
      assert(id_ >=0 && id_ < parent_.qn_);
      int first = parent_.row_ids_[id_](0);
      int second =  parent_.row_ids_[id_](1);
      return std::make_pair(parent_.queries_raw_[first], 
                            parent_.queries_raw_[second]);                    
    }
  };

  Iter begin() const { return Iter(0, *this); }
  Iter end() const {return Iter(qn_, *this);}

  int size() const {return qn_;}
 private:
std::vector<VectorType> queries_raw_;
std::vector<DenseVector<int>> row_ids_;
int qn_;
};

// query iterators for plain vectors.
template <>
class QueryIterator<VectorType> {
public:
  explicit QueryIterator(std::string raw_query_file) :
  queries_raw_(read_data<VectorType>(raw_query_file)),
  qn_(queries_raw_.size()) {}

  QueryIterator() = default;

  
  class Iter {
    int id_;
    const QueryIterator& parent_;
   public:
    explicit Iter(int id, const QueryIterator& parent): id_(id), parent_(parent) {}
    Iter& operator++() {++id_; return *this; }
    bool operator==(const Iter& other) const {return id_ == other.id_; }

    const VectorType& operator*() {
      assert(id_ >=0 && id_ < parent_.qn_);
      return parent_.queries_raw_[id_];                   
    }
    int qid() const {return id_;}
  };

  Iter begin() const { return Iter(0, *this); }
  Iter end() const {return Iter(qn_, *this);}

  int size() const {return qn_;}
 private:
std::vector<VectorType> queries_raw_;
int qn_;
};


template <typename PointType>
auto write_data(std::ofstream& fout, const PointType& point) -> void {
  static_assert(always_false_v<PointType>, "Not supported point type!");
}

template <>
auto write_data<VectorType>(std::ofstream& fout, const VectorType& point) -> void {
  static_assert(std::is_same_v<float&, decltype(std::declval<VectorType>()[0])>);
  int sz = point.size();
  fout.write(reinterpret_cast<char*> (&sz), 4);
  fout.write(reinterpret_cast<const char*> (point.data()), sizeof(float) * sz);
}

template <>
auto write_data<MatrixType>(std::ofstream& fout, const MatrixType& matrix) -> void {
  static_assert(std::is_same_v<float&, decltype(std::declval<MatrixType>()(0, 0))>);
  int num_rows = matrix.rows();
  for (int i = 0; i < num_rows; ++i) {
    VectorType row = matrix.row(i);
    write_data(fout, row);
  }
}

template <typename T>
auto write_data(std::ofstream& fout, const std::vector<T>& point) -> void {
  int sz = point.size();
  fout.write(reinterpret_cast<char*> (&sz), 4);
  std::vector<float> casted_vec(sz);
  for(int i = 0; i < sz; ++i) {
    casted_vec[i] = static_cast<float>(point[i]);
  }
  fout.write(reinterpret_cast<const char*> (casted_vec.data()), sizeof(float) * sz);
}

template <typename PointType>
auto write_double_data(std::ofstream& fout, const PointType& point) -> void {
  static_assert(always_false_v<PointType>, "Not supported point type!");
}

template <typename T>
auto write_double_data(std::ofstream& fout, const std::vector<T>& point) -> void {
  int sz = point.size();
  fout.write(reinterpret_cast<char*> (&sz), 4);
  std::vector<double> casted_vec(sz);
  for(int i = 0; i < sz; ++i) {
    casted_vec[i] = static_cast<double>(point[i]);
  }
  fout.write(reinterpret_cast<const char*> (casted_vec.data()), sizeof(double) * sz);
}


template <typename PointType>
auto write_int_data(std::ofstream& fout, const PointType& point) -> void {
  static_assert(always_false_v<PointType>, "Not supported point type!");
}

template <typename T>
auto write_int_data(std::ofstream& fout, const std::vector<T>& point) -> void {
  int sz = point.size();
  fout.write(reinterpret_cast<char*> (&sz), 4);
  std::vector<int> casted_vec(sz);
  for(int i = 0; i < sz; ++i) {
    casted_vec[i] = static_cast<int>(point[i]);
  }
  fout.write(reinterpret_cast<const char*> (casted_vec.data()), sizeof(int) * sz);
}

std::vector<std::vector<int>> read_candidates(const LSHConstructionParameters &conf) {
  std::vector<std::vector<int>> candidates(conf.num_queries);
  int active_table = conf.hash_table_params[1].l;
  for (int qid = 0; qid < conf.num_queries; ++qid) {
    std::string candidate_file = conf.candidate_filename + "_q_" + std::to_string(qid) + ".bin";
    std::ifstream fin(candidate_file, std::ios::binary);

    unsigned char mode;
    int num_candidates;

    fin.read(reinterpret_cast<char*> (&mode), sizeof(mode));
    bool read_bucket = (mode == 0b11100000);
    assert(mode == 0b11000000 || read_bucket);
    fin.read(reinterpret_cast<char*> (&num_candidates), sizeof(int));
    for (int cid = 0; cid < num_candidates; ++cid) {
      int id, table, bucket;
      fin.read(reinterpret_cast<char*> (&id), sizeof(int));
      fin.read(reinterpret_cast<char*> (&table), sizeof(int));
      if (read_bucket) {
        fin.read(reinterpret_cast<char*> (&bucket), sizeof(int));
      }
      if (table >= active_table) continue;
      candidates[qid].push_back(id);
    }
    std::sort(candidates[qid].begin(), candidates[qid].end());
    auto last_unq = std::unique(candidates[qid].begin(), candidates[qid].end());
    candidates[qid].erase(last_unq, candidates[qid].end());
  }
  return candidates;
}

std::vector<std::vector<int>> read_candidates_by_table(const LSHConstructionParameters &conf, int qid) {
  int num_active_tables = conf.hash_table_params[1].l;
  std::vector<std::vector<int>> candidates(num_active_tables);
  std::string candidate_file = conf.candidate_filename + "_q_" + std::to_string(qid) + ".bin";
  std::ifstream fin(candidate_file, std::ios::binary);

  unsigned char mode;
  int num_candidates;

  fin.read(reinterpret_cast<char*> (&mode), sizeof(mode));
  bool read_bucket = (mode == 0b11100000);
  assert(mode == 0b11000000 || read_bucket);
  fin.read(reinterpret_cast<char*> (&num_candidates), sizeof(int));
  for (int cid = 0; cid < num_candidates; ++cid) {
    int id, table, bucket;
    fin.read(reinterpret_cast<char*> (&id), sizeof(int));
    fin.read(reinterpret_cast<char*> (&table), sizeof(int));
    if (read_bucket) {
      fin.read(reinterpret_cast<char*> (&bucket), sizeof(int));
    }
    if (table >= num_active_tables) break;
    candidates[table].push_back(id);
  }
  return candidates;
}

std::vector<std::vector<int>> read_candidates_by_table_fair(const LSHConstructionParameters &conf, int qid) {
  int num_tables = conf.hash_table_params[0].l;
  std::vector<std::vector<int>> candidates(num_tables);
  std::string candidate_file = conf.candidate_filename + "_q_" + std::to_string(qid) + ".bin";
  std::ifstream fin(candidate_file, std::ios::binary);


  unsigned char mode;
  int num_candidates;

  fin.read(reinterpret_cast<char*> (&mode), sizeof(mode));
  bool read_bucket = (mode == 0b11100000);
  assert(mode == 0b11000000 || read_bucket);
  fin.read(reinterpret_cast<char*> (&num_candidates), sizeof(int));
  for (int cid = 0; cid < num_candidates; ++cid) {
    int id, table, bucket;
    fin.read(reinterpret_cast<char*> (&id), sizeof(int));
    fin.read(reinterpret_cast<char*> (&table), sizeof(int));
    if (read_bucket) {
      fin.read(reinterpret_cast<char*> (&bucket), sizeof(int));
    }
    if (table >= num_tables) break;
    candidates[table].push_back(id);
  }
  return candidates;
}

template <typename T>
std::string vec_to_string(const std::vector<T>& vec) {
  std::string result = "[";
  for (const auto& v : vec) {
    result += std::to_string(v) + ", ";
  }
  result += "]";
  return result;
}

inline std::ofstream open_file_create(const std::filesystem::path& path) {
  std::ofstream fout(path);
  // if path is empty, do not output
  if (!fout && path != "") {
    try {
      std::filesystem::create_directories(path.parent_path());
    } catch (std::filesystem::filesystem_error& e) {
      std::cerr << e.what() << std::endl;
      std::cerr << "Failed to create directory: " << path.parent_path() << std::endl;
      throw e;
    }
    fout.close();
    fout.open(path);
  }
  return fout;
}

} // namespace falconn

#endif