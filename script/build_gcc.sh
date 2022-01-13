#!/bin/sh

set -eu

GCC_COMMAND=${C_COMPILER:-"gcc"}
GXX_COMMAND=${CXX_COMPILER:-"g++"}
OPT_FLAGS=${QULACS_OPT_FLAGS:-"-mtune=native -march=native -mfpmath=both"}
# Tell the Python interpreter path to pybind.
PYTHON_EXECUTABLE=$(python -c "import sys; print(sys.executable)")

mkdir -p ./build
cd ./build
cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D OPT_FLAGS="$OPT_FLAGS" -D CMAKE_BUILD_TYPE=Release USE_GPU:STR=No -D PYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" ..
make -j $(nproc)
make python -j $(nproc)
cd ../
