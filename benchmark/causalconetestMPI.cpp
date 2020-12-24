#include <vector>
#include <utility>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/type.hpp>
#include "../src/mpisim/causal_coneMPI.hpp"
#include <mpi.h>
using namespace std;
int main(){
	MPI::Init();
	const UINT n = 8;
	const UINT depth = 5;
	QuantumCircuit circuit(n * 2);
	Observable observable(n * 2);
	int rank = MPI::COMM_WORLD.Get_rank();
	QuantumState state(n * 2);
	state.set_zero_state();
	for(int i = 0; i < depth; i++){
		for(int j = 0; j < n * 2; j++){
			circuit.add_RX_gate(j, 3.1415 * 0.25);
			circuit.add_RZ_gate(j, 3.1415 * 0.25);
		}
		for(int j = 0; j < n; j++){
			circuit.add_CNOT_gate(j * 2, j * 2 +1);
		}
		for(int j = 0; j < n; j++){
			circuit.add_CNOT_gate(j * 2 + 1, (j * 2 + 2) % (n * 2));
		}
	}
	circuit.update_quantum_state(&state);
	std::string Pauli_string = "";
  for(int i = 0;i < n * 2;i++){
		std::string Pauli_string = "Z ";
		Pauli_string += std::to_string(i);
		observable.add_operator(5.0, Pauli_string.c_str());
	}
	Causal c;
	auto ret = c.CausalConeMPI(circuit, observable);
	CPPCTYPE sum;
	MPI::COMM_WORLD.Allreduce(&ret, &sum, 1, MPI::DOUBLE_COMPLEX,MPI::SUM);
	cout<<rank<<" "<<ret<<"\n";
	if(rank == 0){
		std::cout<<sum<<" " << observable.get_expectation_value(&state) << "\n";
	}

	MPI::Finalize();
}
