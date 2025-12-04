#!/bin/bash

#SBATCH -J system_qulacs            
#SBATCH -o system_qulacs_%j.o       
#SBATCH -e system_qulacs_%j.e       
# SBATCH -N 1                
#SBATCH -c 1                
#SBATCH -n 1        
#SBATCH -t 00:10:00         
#SBATCH --mem-per-cpu=15G    

ml qmio/hpc gcc/12.3.0 qulacs/0.6.3-python-3.9.9-mpi

#mpirun python -u system_qulacs_test.py 

srun ./proba_unroll