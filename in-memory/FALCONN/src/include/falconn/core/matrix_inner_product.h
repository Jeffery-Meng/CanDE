#ifndef __MATRIX_INNER_PRODUCT_H__
#define __MATRIX_INNER_PRODUCT_H__

#include <cstdint>
#include <vector>
#include "../falconn_global.h"
#include <Eigen/Dense>

namespace falconn {
namespace core {


template <typename DataType, typename QueryType>
struct MatrixInnerProduct {};

template <>
struct MatrixInnerProduct<VectorType, VectorPairType> {
  CoordinateType operator()(const VectorPairType& query, const VectorType& data) {
    CoordinateType inner_p1 = data.transpose() * query.first;
    CoordinateType inner_p2 = data.transpose() * query.second;
    return inner_p1 * inner_p2;
  }
};

template <>
struct MatrixInnerProduct<VectorPairType, VectorPairType> {
  CoordinateType operator()(const VectorPairType& query,
                            const VectorPairType& data) {
    CoordinateType inner_p1 = data.first.transpose() * query.first;
    CoordinateType inner_p2 = data.second.transpose() * query.second;
    return inner_p1 * inner_p2;
  }
};

template <>
struct MatrixInnerProduct<VectorType, MatrixType> {
  CoordinateType operator()(const MatrixType& query,
                            const VectorType& data) {
    return data.transpose() * query * data;
  }
};

template <>
struct MatrixInnerProduct<VectorPairType, MatrixType> {
  CoordinateType operator()(const MatrixType& query, const VectorPairType& data) {
    return data.first.transpose() * query * data.second;
  }
};


}  // namespace core
}  // namespace falconn

#endif
