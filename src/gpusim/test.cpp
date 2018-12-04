#include "util.h"
#include "update_ops_cuda.h"
#include <complex>
#include <iostream>

int main(){
    unsigned int n_qubits=5;
    ITYPE len=1ULL<<n_qubits;
	void* state = allocate_quantum_state_host(len);
	initialize_quantum_state_host(state, len);
    X_gate_host(0,state,len);
    print_quantum_state_host(state, len);
    X_gate_host(1,state,len);
    print_quantum_state_host(state, len);
    CZ_gate_host(0, 1, state, len);
    print_quantum_state_host(state, len);
    CNOT_gate_host(0, 1, state, len);
    print_quantum_state_host(state, len);
    release_quantum_state_host(state);
	return 0;
}
