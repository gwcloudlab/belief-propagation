project(example-belief-propagation)

cmake_minimum_required(VERSION 2.8)

add_executable(example-belief-propagation example-belief-propagation.c ../graph/graph.c)
target_link_libraries(example-belief-propagation m)