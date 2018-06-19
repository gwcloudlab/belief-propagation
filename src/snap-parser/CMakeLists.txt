project(test-snap-parser)

cmake_minimum_required(VERSION 2.8)

add_executable(test-snap-parser main.c snap-parser.c ../graph/graph.c)