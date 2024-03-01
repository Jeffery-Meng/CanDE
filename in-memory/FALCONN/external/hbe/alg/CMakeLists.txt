cmake_minimum_required(VERSION 3.5)

#set(EIGEN3_DIR "../eigen/Eigen")
#find_package (Eigen3 3.3 REQUIRED)

set(Boost_USE_MULTITHREADED ON)
find_package( Boost REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )

include_directories(../utils)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter")
add_library (alg RS.h RS.cpp Herding.h KCenter.h AdaptiveRSDiag.h AdaptiveRSDiag.cpp naiveKDE.h naiveKDE.cpp RS.h RS.cpp UniformHBE.cpp UniformHBE.h HashBucket.h HashTable.h MoMEstimator.h SketchHBE.cpp SketchHBE.h SketchTable.h AdaptiveEstimator.h AdaptiveRS.cpp AdaptiveRS.h AdaptiveHBE.cpp AdaptiveHBE.h)
target_include_directories(alg PUBLIC "../eigen/Eigen")


#target_link_libraries(alg Eigen3::Eigen)

