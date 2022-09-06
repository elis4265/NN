cmake_minimum_required(VERSION 3.20)
project(nnets)

set(CMAKE_CXX_STANDARD 20)

add_executable(nnets src/main.cpp)
add_executable(nnets_example_xor src/example_xor.cpp)
add_executable(nnets_example_xor_train src/example_xor_train.cpp)
