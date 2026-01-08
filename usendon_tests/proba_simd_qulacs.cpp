

#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>


int main() {
    QuantumState state(2); 

    std::vector<CPPCTYPE> vec = {0.0, 1.0, 2.0, 3.0};
    state.load(vec.data());

    std::cout << "Estado inicial:" << std::endl;
    std::cout << state << std::endl;

    QuantumCircuit circuit(2); // crea un circuito cuántico vacío de 2 qubits

    circuit.add_SWAP_gate(0,1);

    circuit.update_quantum_state(&state); // aplica el circuito al estado cuántico

    std::cout << "Estado después de aplicar el circuito:" << std::endl;
    std::cout << state << std::endl;



    return 0;
}
