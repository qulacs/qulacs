#!/bin/sh

GCC_COMMAND=gcc
GXX_COMMAND=g++

# if gcc/g++ version is less than 8, use gcc-8/g++-8
GCC_VERSION=$(gcc -dumpfullversion -dumpversion | awk -F. '{printf "%2d%02d%02d", $1,$2,$3}')
if [ "$GCC_VERSION" -lt 80000 ]; then
  GCC_COMMAND=gcc-8
fi
GXX_VERSION=$(g++ -dumpfullversion -dumpversion | awk -F. '{printf "%2d%02d%02d", $1,$2,$3}')
if [ "$GXX_VERSION" -lt 80000 ]; then
  GXX_COMMAND=g++-8
fi

mkdir ./build
cd ./build
cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D CMAKE_BUILD_TYPE=Release ..
make
make python
cd ../

