#include "util.h"
#include "update_ops_cuda.h"
#include <complex>
#include <iostream>
#include "memory_ops.h"
#include "stat_ops.h"

int main(int argc, char* argv[]){
    unsigned int n_qubits=2;
    if(argc==2) n_qubits = atoi(argv[1]);
    ITYPE len=1ULL<<n_qubits;
	void* state = allocate_quantum_state_host(len);
	initialize_quantum_state_host(state, len);
    X_gate_host(0,state,len);
    //print_quantum_state_host(state, len);
    X_gate_host(1,state,len);
    //print_quantum_state_host(state, len);
    //CZ_gate_host(0, 1, state, len);
    //print_quantum_state_host(state, len);
    //CNOT_gate_host(0, 1, state, len);
    //print_quantum_state_host(state, len);
    initialize_Haar_random_state_host(state, len);
    printf("haar random state\n");
    //print_quantum_state_host(state, len);

    double norm = state_norm_host(state, len);
    printf("norm: %.8f\n", norm);
    if(n_qubits<=15){
        norm = state_norm_cublas_host(state, len);
        printf("norm answer: %.8f\n", norm);
    }
    release_quantum_state_host(state);
	return 0;
}
