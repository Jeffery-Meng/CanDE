#ifndef __ONIAK_GLOBAL__
#define __ONIAK_GLOBAL__

#include <fstream>
#include "../constexpr.h"

#include "Eigen/Dense"

namespace ONIAK {

// Change DType here affects the global data type in this project.
using DType = falconn::CoordinateType;

using DMatrix = Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;    

using DRowVector = Eigen::Matrix<DType, 1, Eigen::Dynamic, Eigen::RowMajor>;    

using DColumnVector = Eigen::Matrix<DType, Eigen::Dynamic, 1, Eigen::RowMajor>;    

}  // namespace ONIAK

#endif  // __ONIAK_GLOBAL__
