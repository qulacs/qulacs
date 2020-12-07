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
#include "../src/cppsim/causal_cone.hpp"

using namespace std;
int main(){
	const UINT depth = 6;
	for(UINT n = 1; n <= 13; n++){
		QuantumState state(n * 2);
		QuantumCircuit circuit(n * 2);
		for(UINT i = 0; i < depth; i++){
			for(UINT j = 0; j < n; j++){
				std::vector<UINT> v = {j * 2, j * 2 + 1};
				circuit.add_random_unitary_gate(v);
			}
		}
		Observable observable(n * 2);
		mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
		for(UINT i = 0; i < (n * 2); i++){
			string s; s+="Z "; s+= to_string(i);
			observable.add_operator(rnd(), s);
		}

		auto start = std::chrono::system_clock::now();
		circuit.update_quantum_state(&state);
		auto value = observable.get_expectation_value(&state);
		auto end = std::chrono::system_clock::now();

		double t1 = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		start = std::chrono::system_clock::now();
		auto v2= CausalCone(circuit, observable);
		end = std::chrono::system_clock::now();
		double t2 = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
		//cout<<"value"<<" "<<value<<" "<<v2<<"\n";
		cout<< n * 2 <<" "<<depth * 2 <<" "<<t1<<" "<<t2<<"\n";
	}
}