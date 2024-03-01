cmake_minimum_required(VERSION 3.0)

project(simulator)

add_library(data_generator
        "data_generator.h"
        "data_generator.cc")

target_include_directories(data_generator PUBLIC
                          "../../external")

