#!/bin/sh

set -eux

GCC_COMMAND=${C_COMPILER:-"gcc"}
GXX_COMMAND=${CXX_COMPILER:-"g++"}

USE_TEST=${USE_TEST:-"No"}
USE_GPU=${USE_GPU:-"No"}
COVERAGE=${COVERAGE:-"No"}

CMAKE_ALL_OPS="-G \"Unix Makefiles\" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D CMAKE_BUILD_TYPE=Release"
CMAKE_ALL_OPS="$CMAKE_ALL_OPS -D USE_GPU=${USE_GPU}"
CMAKE_ALL_OPS="$CMAKE_ALL_OPS -D USE_TEST=${USE_TEST}"
CMAKE_ALL_OPS="$CMAKE_ALL_OPS -D COVERAGE=${COVERAGE}"

mkdir -p ./build
cd ./build
if [ "${QULACS_OPT_FLAGS:-"__UNSET__"}" = "__UNSET__" ]; then
  cmake -G "Unix Makefiles" $CMAKE_OPTIONS ..
else
  cmake -G "Unix Makefiles" $CMAKE_OPTIONS -D OPT_FLAGS="${QULACS_OPT_FLAGS}" ..
fi
cmake $CMAKE_ALL_OPS ..
make -j $(nproc)
cd ../
