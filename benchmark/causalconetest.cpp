#include <chrono>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>
#include <cppsim/type.hpp>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <vqcsim/parametric_circuit.hpp>

#include "../src/vqcsim/causalcone_simulator.hpp"

using namespace std;
int main() {
    const UINT n = 13;
    const UINT depth = 3;
    QuantumState state(n * 2);
    ParametricQuantumCircuit circuit(n * 2);
    for (UINT i = 0; i < depth; i++) {
        for (UINT j = 0; j < n; j++) {
            std::vector<UINT> v = {j * 2, j * 2 + 1};
            circuit.add_random_unitary_gate(v);
        }
        for (UINT j = 0; j < n; j++) {
            std::vector<UINT> v = {(j * 2 + 1) % 2, (j * 2 + 2) % 2};
            circuit.add_random_unitary_gate(v);
        }
    }
    Observable observable(n * 2);
    mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
    for (int i = 0; i < (int)n * 2; ++i) {
        std::string s = "Z ";
        s += std::to_string(i);
        observable.add_operator(rnd(), s);
    }
    circuit.update_quantum_state(&state);
    auto value = observable.get_expectation_value(&state);
    CausalConeSimulator c(circuit, observable);
    auto value2 = c.get_expectation_value();
    cout << value << " " << value2 << endl;
}