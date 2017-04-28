cmake_minimum_required(VERSION 3.7)
project(cpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GTK_CFLAGS} -O3")


set(SOURCE_FILES main.cpp math_functions.h data.h definitions.h RandomFuzzyForest.h RandomFuzzyTree.h includes.h test_RandomForest.h)
add_executable(cpp ${SOURCE_FILES})