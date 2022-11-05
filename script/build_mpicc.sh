#!/bin/sh

export C_COMPILER=mpicc
export CXX_COMPILER=mpic++
export USE_GPU=No
export USE_MPI=Yes
export USE_TEST=${USE_TEST:-"No"}

./script/build_gcc.sh
