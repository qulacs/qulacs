#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>


int main() {
    QuantumState state(3); // crea un estado cuántico de 2 qubits en |00⟩

    std::cout << "Estado inicial |00>:" << std::endl;
    std::cout << state << std::endl;

    QuantumCircuit circuit(3); // crea un circuito cuántico vacío de 2 qubits

    circuit.add_CNOT_gate(0,1);

     X_gate(0, state.data_c(), state.dim);

    //circuit.add_X_gate(0); // añade una puerta X al qubit 0

    //circuit.add_RX_gate(1, 0.5); // añade una rotación RX al qubit 1

    circuit.update_quantum_state(&state); // aplica el circuito al estado cuántico

    std::cout << "Estado después de aplicar el circuito:" << std::endl;
    std::cout << state << std::endl;



    return 0;
}
