nvcc -O3 -arch sm_30 -lcublas -shared -o libgpusim.so update_ops_*.cu util.cu memory_ops.cu stat_ops.cu
nvcc test.cpp -llibgpusim -o a.out
./a.out

#nvcc QCsim.cu update_ops_*.cu -shared -o libgpusim.so
#g++ test.cpp libgpusim.so -o a.out
#./a.out

