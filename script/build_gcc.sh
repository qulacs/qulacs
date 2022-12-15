#!/bin/sh

set -eux

GCC_COMMAND=${C_COMPILER:-"gcc"}
GXX_COMMAND=${CXX_COMPILER:-"g++"}

mkdir -p ./build
cd ./build
if [ "${QULACS_OPT_FLAGS:-"__UNSET__"}" = "__UNSET__" ]; then
  cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D CMAKE_BUILD_TYPE=Release -D USE_GPU:STR=No -D USE_TEST="${USE_TEST:-No}" -D COVERAGE="${COVERAGE:-No}" ..
else
  cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D OPT_FLAGS="${QULACS_OPT_FLAGS}" -D CMAKE_BUILD_TYPE=Release -D USE_GPU:STR=No -D USE_TEST="${USE_TEST:-No}" -D COVERAGE="${COVERAGE:-No}" ..
fi
make -j $(nproc)
cd ../
