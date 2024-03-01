cmake_minimum_required(VERSION 3.0)

project(falconn-test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXXFLAGS "-O3")
include_directories(../include/falconn)
add_subdirectory(simulation)


add_executable(multiprobe_test "multiprobe_test.cc")
target_link_libraries(multiprobe_test PUBLIC data_generator kronecker_lsh)

add_executable(polytope_test "polytope_test.cc")
target_link_libraries(polytope_test PUBLIC data_generator kronecker_lsh)

add_executable(mp_score_test "mp_score_test.cc")
target_link_libraries(mp_score_test PUBLIC data_generator kronecker_lsh)

add_executable(mp_score_test2 "mp_score_test2.cc")
target_link_libraries(mp_score_test2 PUBLIC data_generator kronecker_lsh)

add_executable(gt_time "gt_time.cc")
target_link_libraries(gt_time PUBLIC data_generator kronecker_lsh)

add_executable(mp_collision "mp_collision.cc")
target_link_libraries(mp_collision PUBLIC data_generator kronecker_lsh)

add_executable(compact_bool_test "compact_bool_test.cc")