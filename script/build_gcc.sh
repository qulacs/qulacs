#!/bin/sh

set -eu

GCC_COMMAND=${C_COMPILER:-"gcc"}
GXX_COMMAND=${CXX_COMPILER:-"g++"}
OPT_FLAGS=${QULACS_OPT_FLAGS:-"-mtune=native -march=native -mfpmath=both"}

mkdir -p ./build
cd ./build
cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D OPT_FLAGS="$OPT_FLAGS" -D CMAKE_BUILD_TYPE=Release USE_GPU:STR=No ..
make -j $(nproc)
cd ../
