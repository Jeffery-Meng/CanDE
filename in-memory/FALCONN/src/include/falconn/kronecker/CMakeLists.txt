cmake_minimum_required(VERSION 3.0)

project(lsh)

set(rotation_hdr
        "cross_polytope_lsh.h"
        "kronecker_lsh.h"
        "pseudorandom_rotation.h")
set(rotation_src
        "cross_polytope_lsh.cc"
        "kronecker_lsh.cc"
        "pseudorandom_rotation.cc")

add_library(kronecker_lsh
        ${rotation_hdr}
        ${rotation_src})

target_link_libraries(kronecker_lsh PUBLIC ffht)


target_include_directories(kronecker_lsh PUBLIC
                          "../../external" "../ffht")





