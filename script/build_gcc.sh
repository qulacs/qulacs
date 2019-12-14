#!/bin/sh

GCC_COMMAND="$C_COMPILER"
GXX_COMMAND="$CXX_COMPILER"

if [ -z "$GCC_COMMAND" ]; then
  GCC_COMMAND="gcc"
fi

if [ -z "$GXX_COMMAND" ]; then
  GXX_COMMAND="g++"
fi

# if gcc/g++ version is less than 8, use gcc-8/g++-8
GCC_VERSION=$($GCC_COMMAND -dumpfullversion -dumpversion | awk -F. '{printf "%2d%02d%02d", $1,$2,$3}')
if [ "$GCC_VERSION" -lt 80000 ]; then
  GCC_COMMAND=gcc-7
elif [ "$GCC_VERSION" -lt 90000 ]; then
  GCC_COMMAND=gcc-8
fi

GXX_VERSION=$($GXX_COMMAND -dumpfullversion -dumpversion | awk -F. '{printf "%2d%02d%02d", $1,$2,$3}')
if [ "$GXX_VERSION" -lt 80000 ]; then
  GXX_COMMAND=g++-7
elif [ "$GXX_VERSION" -lt 90000 ]; then
  GXX_COMMAND=g++-8
fi

mkdir ./build
cd ./build
cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D CMAKE_BUILD_TYPE=Release ..
make
make python
cd ../

