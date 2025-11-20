#!/bin/sh

ml load qmio/hpc gcc/12.3.0 impi/2021.13.0 boost/1.85.0 cmake/3.27.6 python/3.9.9

if [[ -z "$1" || "$1" != "shared" ]]; then
    echo "Static compilation"

    source script/build_mpicc.sh

    g++ -O2 -I./include/ -L./lib/ -Wl,-rpath,./lib/ \
                        proba_ecr.cpp -o proba_ecr \
                        -lcppsim_static -lcsim_static -lvqcsim_static -lmpi \
                        -fopenmp -D_USE_MPI

else
    echo "Shared compilation"

    source script/build_mpicc.sh
    cd build/
    make shared
    cd ../

    g++ -O2 -I./include/ -L./build/src/cppsim -Wl,-rpath,./build/src/cppsim \
                         -L./build/src/csim -Wl,-rpath,build/src/csim \
                         -L./build/src/vqcsim -Wl,-rpath,./build/src/vqcsim \
                         proba_ecr.cpp -o proba_ecr \
                         -lcppsim_shared -lcsim_shared -lvqcsim_shared -lmpi \
                         -fopenmp -D_USE_MPI
    
fi