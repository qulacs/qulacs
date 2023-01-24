#!/bin/sh

set -eux

GCC_COMMAND=clang
GXX_COMMAND=clang++

mkdir -p ./build
cd ./build
cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D CMAKE_BUILD_TYPE=Release -D USE_GPU:STR=No -D USE_TEST="${USE_TEST:-No}" -D USE_OMP:STR=No ..
make -j $(nproc)
cd ../
