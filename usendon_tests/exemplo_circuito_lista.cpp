#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>


int main() {
    QuantumState state(3); // crea un estado cuántico de 3 qubits en |000⟩

    std::cout << "Estado inicial |000>:" << std::endl;
    std::cout << state << std::endl;

    /* state.set_Haar_random_state();  // cambia el estado a uno aleatorio (distribución de Haar)

    std::cout << "Estado aleatorio (Haar):" << std::endl;
    std::cout << state << std::endl; */

    QuantumCircuit circuit(3); // crea un circuito cuántico vacío de 3 qubits

    circuit.add_H_gate(0); // añade una puerta X al qubit 0

    circuit.add_CNOT_gate(0, 1); // añade una rotación RX al qubit 1

    circuit.update_quantum_state(&state); // aplica el circuito al estado cuántico

    std::cout << "Estado después de aplicar el circuito:" << std::endl;
    std::cout << state << std::endl;

    return 0;
}