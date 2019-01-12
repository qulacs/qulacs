#!/bin/sh

GCC_COMMAND=gcc
GXX_COMMAND=g++

# if gcc/g++ version is less than 7, use gcc-7/g++-7
GCC_VERSION=$(gcc -dumpfullversion -dumpversion | awk -F. '{printf "%2d%02d%02d", $1,$2,$3}')
if [ "$GCC_VERSION" -lt 70000 ]; then
  GCC_COMMAND=gcc-7
fi
GXX_VERSION=$(g++ -dumpfullversion -dumpversion | awk -F. '{printf "%2d%02d%02d", $1,$2,$3}')
if [ "$GXX_VERSION" -lt 70000 ]; then
  GXX_COMMAND=g++-7
fi

mkdir ./build
cd ./build
cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D CMAKE_BUILD_TYPE=Release -D USE_GPU:STR=Yes ..
make
make python
cd ../

