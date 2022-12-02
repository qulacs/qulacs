#!/bin/sh

set -eux

GCC_COMMAND=${C_COMPILER:-"gcc"}
GXX_COMMAND=${CXX_COMPILER:-"g++"}

USE_TEST=${USE_TEST:-"No"}
USE_GPU=${USE_GPU:-"No"}
USE_MPI=${USE_MPI:-"No"}
COVERAGE=${COVERAGE:-"No"}

CMAKE_OPS="-D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D CMAKE_BUILD_TYPE=Release"
CMAKE_OPS="${CMAKE_OPS} -D USE_MPI=${USE_MPI} -D USE_GPU=${USE_GPU}"
CMAKE_OPS="${CMAKE_OPS} -D USE_TEST=${USE_TEST} -D COVERAGE=${COVERAGE}"

mkdir -p ./build
cd ./build
if [ "${QULACS_OPT_FLAGS:-"__UNSET__"}" = "__UNSET__" ]; then
  cmake -G "Unix Makefiles" ${CMAKE_OPS} ..
else
  cmake -G "Unix Makefiles" ${CMAKE_OPS} -D OPT_FLAGS="${QULACS_OPT_FLAGS}" ..
fi
make -j $(nproc)
cd ../
