#!/bin/bash

#SBATCH -J qulacs_mpi            
#SBATCH -o qulacs_mpi_%j.o       
#SBATCH -e qulacs_mpi_%j.e       
# SBATCH -N 1                
#SBATCH -c 1                
#SBATCH -n 8     
#SBATCH -t 00:10:00         
#SBATCH --mem-per-cpu=15G    

ml load qmio/hpc gcc/12.3.0 impi/2021.13.0 boost/1.85.0 cmake/3.27.6 python/3.9.9

mpirun ./mpi_test