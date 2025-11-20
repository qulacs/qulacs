#!/bin/sh

export C_COMPILER=mpicc
export CXX_COMPILER=mpicxx # Antes mpic++
export USE_GPU=No
export USE_MPI=Yes

./script/build_gcc.sh
