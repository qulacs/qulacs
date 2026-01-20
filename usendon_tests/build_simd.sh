#!/bin/sh

ml load qmio/hpc gcc/12.3.0  gcccore/12.3.0 impi/2021.13.0 boost/1.85.0 cmake/3.27.6 nlohmann_json/3.11.3 python/3.9.9

if [[ -z "$1" || "$1" != "shared" ]]; then
    echo "Static compilation"

    cd ..
    source script/build_gcc.sh
    cd usendon_tests

    g++ -O2 -I../include/ -L../lib/ -Wl,-rpath,../lib/ \
                        ./proba_simd_swap.cpp -o proba_simd \
                        -lcppsim_static -lcsim_static -lvqcsim_static -lmpi \
                        -fopenmp 

else
    echo "Shared compilation"

    cd ..
    source /script/build_gcc.sh
    cd build/
    make shared
    cd ../usendon_tests

    g++ -O2 -I../include/ -L../build/src/cppsim -Wl,-rpath,../build/src/cppsim \
                         -L../build/src/csim -Wl,-rpath,../build/src/csim \
                         -L../build/src/vqcsim -Wl,-rpath,../build/src/vqcsim \
                         ./proba_simd_swap.cpp -o proba_simd \
                         -lcppsim_shared -lcsim_shared -lvqcsim_shared -lmpi \
                         -fopenmp 
    
fi