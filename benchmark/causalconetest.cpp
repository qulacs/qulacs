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
	const UINT n = 5;
	const UINT depth = 3;
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
	string s = "Z " + to_string(0) + "Z " + to_string(n);
	observable.add_operator(rnd(), s);
	circuit.update_quantum_state(&state);
	auto value = observable.get_expectation_value(&state);
	auto value2 = CausalCone(circuit, observable);
	cout<<value<<" "<<value2<<endl;

}