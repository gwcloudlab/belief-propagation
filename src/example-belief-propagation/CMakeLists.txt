cmake_minimum_required(VERSION 2.8)

project(example-belief-propagation)
add_executable(example-belief-propagation example-belief-propagation.c ../graph/graph.c ../node/node.c ../edge/edge.c)