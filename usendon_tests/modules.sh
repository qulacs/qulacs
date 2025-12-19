#!/bin/bash

ml load qmio/hpc gcc/12.3.0  gcccore/12.3.0 impi/2021.13.0 boost/1.85.0 cmake/3.27.6 nlohmann_json/3.11.3 python/3.9.9

# Modulos ARM
#ml gnu12/12.2.0 openmpi4/4.1.4 boost/1.80.0 cmake/3.24.2 

# Flags que teño que incluír para compilar un arquivo no nodo de ARM
#-I/opt/cesga/qmio/hpc/software/Compiler/gcccore/12.3.0/nlohmann_json/3.11.3/include/ 
#-Wl,-rpath, /opt/cesga/qmio/hpc/software/Compiler/gcccore/12.3.0/nlohmann_json/3.11.3/include/

#g++ -O2 -I ../include -L ../lib  proba_unroll.cpp -o proba_unroll -lvqcsim_static -lcppsim_static -lcsim_static -fopenmp -D_USE_SVE -I/opt/cesga/qmio/hpc/software/Compiler/gcccore/12.3.0/nlohmann_json/3.11.3/include/ -Wl,-rpath, /opt/cesga/qmio/hpc/software/Compiler/gcccore/12.3.0/nlohmann_json/3.11.3/include/


