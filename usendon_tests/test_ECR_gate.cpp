#include <iostream>

#include <cppsim/circuit.hpp>
#include "cppsim/simulator.hpp"



int main() {

    QuantumCircuit circuit(2);
    //circuit.add_X_gate(0);
    circuit.add_ECR_gate(0, 1);
    

    QuantumCircuitSimulator sim(&circuit);
    UINT n_gates = sim.get_gate_count();

    std::cout << "Number of gates: " << n_gates << "\n";


    return 0;

}