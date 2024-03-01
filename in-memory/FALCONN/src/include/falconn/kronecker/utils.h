#ifndef __UTILS_H__
#define __UTILS_H__

#include <bitset>
#include <cassert>
#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <Eigen/Dense>
#include <numeric>
#include "boost/unordered_map.hpp"

namespace ONIAK {

// returns whether v is a power of 2.
inline bool IsPowerOfTwo(int v) {
  return (v & (v - 1)) == 0;
}

inline int LogOfTwo(int v) {
  return 8 * sizeof(int) - __builtin_clz(v) - 1;
}



template<typename T>
std::ostream& operator<<(std::ostream& os, const Eigen::Matrix<T, 1, Eigen::Dynamic>& row){
  for (auto val: row) {
    os << val << "\t";
  }
  return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& row){
  for (auto val: row) {
    os << val << "\t";
  }
  return os;
}


template<typename T>
std::ostream& operator<<(std::ostream& os, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix){
  int rows = matrix.rows();
  for (int rr = 0; rr < rows; ++rr) {
    os << matrix.row(rr) << std::endl;
  }
  return os;
}

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T>
using EigenRowVector = Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T = float>
EigenMatrix<T> read_one_matrix(std::ifstream& fin, int rows, int cols) {
  size_t type_sz = sizeof(T);
  int sz;
  EigenMatrix<T> result(rows, cols);
    for (int rr = 0; rr < rows; ++rr) {
      fin.read(reinterpret_cast<char*> (&sz), 4);
      assert(sz == cols && "cols does not agree with file.");
      fin.read(reinterpret_cast<char*> (&result(rr, 0)), type_sz * cols);
      assert(fin && "EOF error.");
    }
    return result;
}

template <typename T = float>
EigenRowVector<T> read_vector(std::ifstream& fin) {
  size_t type_sz = sizeof(T);
  int sz;
  fin.read(reinterpret_cast<char*> (&sz), 4);
  EigenRowVector<T> result(sz);
  fin.read(reinterpret_cast<char*> (result.data()), type_sz * sz);
  assert(fin && "EOF error.");
  return result;
}

template <typename T = float>
EigenMatrix<T> read_file(std::string filename) {
  size_t type_sz = sizeof(T);
  size_t file_sz = std::filesystem::file_size(filename);
  std::ifstream fin(filename);
  int sz;
  fin.read(reinterpret_cast<char*> (&sz), 4);
  int64_t cols = sz, rows = file_sz / (cols * type_sz + sizeof(int));
  assert(cols * rows * type_sz + rows * sizeof(int) == file_sz);
  EigenMatrix<T> result(rows, cols);
  for (int rr = 0; rr < rows; ++rr) {
    fin.read(reinterpret_cast<char*> (&result(rr, 0)), type_sz * cols);
    fin.read(reinterpret_cast<char*> (&sz), 4);
  }
  return result;
}

template <typename T, typename FT = float>
void write_file(std::ofstream& fout, const EigenMatrix<T>& input) {
  EigenMatrix<FT> converted_matrix;
  const EigenMatrix<FT>* cview;
  if constexpr (std::is_same_v<T, FT>) {
    cview = &input;
  } else {
    converted_matrix = input.template cast<FT>();
    cview = &converted_matrix;
  }
  const EigenMatrix<FT>& matrix = *cview;

  size_t type_sz = sizeof(FT);
  int sz = matrix.cols();
  for (int rr = 0; rr < matrix.rows(); ++rr) {
    fout.write(reinterpret_cast<char*> (&sz), 4);
    fout.write(reinterpret_cast<const char*> (&matrix(rr, 0)), type_sz * sz);
  }
}

/* Return orders of input containter
The i-th entry of output is the index of the i-th smallest item in input.
For example, if input is {1, 5, 3, 2, 4}, then output is {0, 3, 2, 4, 1}*/
template <typename IterType>
std::vector<size_t> Orders(IterType begin, IterType end) {
  int N = end - begin;
  std::vector<size_t> V(N);
  std::iota(V.begin(),V.end(),0); //Initializing
  std::sort( V.begin(),V.end(),  [&](int i,int j){return *(begin + i)<*(begin + j);});
  return V;
}

template <typename IterType, typename Comp>
std::vector<size_t> Orders(IterType begin, IterType end, 
                Comp comp) {
  int N = end - begin;
  std::vector<size_t> V(N);
  std::iota(V.begin(),V.end(),0); //Initializing
  std::sort( V.begin(),V.end(), comp);
  return V;
}

// The first k elements are largest
template <typename IterType>
std::vector<size_t> kLargestIndexes(IterType begin, IterType end, int k) {
  int N = end - begin;
  std::vector<size_t> V(N);
  assert(k <= N);
  std::iota(V.begin(),V.end(),0); //Initializing
  std::nth_element( V.begin(), V.begin() + k, V.end(), [&](int i,int j){return std::abs(*(begin + i)) > std::abs(*(begin + j));} );
  return V;
}

template<typename T>
void print_one_line(std::ostream& os, T value) {
    os << value << std::endl;
}

template<typename T, typename... Targs>
void print_one_line(std::ostream& os, T value, Targs... Fargs) {
    os << value << "\t";
    print_one_line(os, Fargs...);
}

// Prints table in csv format.
template<typename T>
void print_table(std::ostream& os, T value) {
    os << value << std::endl;
}

template<typename T, typename... Targs>
void print_table(std::ostream& os, T value, Targs... Fargs) {
    os << value << ",";
    print_table(os, Fargs...);
}

template<typename T, size_t N>
std::bitset<N> vector_to_bitset(const std::vector<T>& vector) {
  std::bitset<N> bitset;
  for (T val : vector) {
    bitset.set(val);
  }
  return bitset;
}

template<typename KT, typename T>
std::ostream& operator<<(std::ostream& os, const boost::unordered_map<KT, T>& hashmap){
  for (auto item : hashmap) {
    os << "(" << item.first << ", " << item.second << ")\t";
  }
  os << std::endl;
  return os;
}

} // namespace ONIAK

#endif // __UTILS_H__