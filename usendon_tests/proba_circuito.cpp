
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

    state.set_Haar_random_state();  // cambia el estado a uno aleatorio (distribución de Haar)

    std::cout << "Estado aleatorio (Haar):" << std::endl;
    std::cout << state << std::endl;

    QuantumCircuit circuit(3); // crea un circuito cuántico vacío de 3 qubits

    std::cout << "Circuito inicial:" << std::endl;
    std::cout << circuit << std::endl;

    circuit.add_X_gate(0); // añade una puerta X al qubit 0

    std::cout << "Circuito tras X en qubit 0:" << std::endl;
    std::cout << circuit << std::endl;

    auto merged_gate = gate::merge(gate::CNOT(0, 1), gate::Y(1));
    // crea una puerta fusionada: primero CNOT(0,1), luego Y(1)

    std::cout << "Puerta fusionada (CNOT + Y):" << std::endl;
    std::cout << *merged_gate << std::endl;

    circuit.add_gate(merged_gate);

    std::cout << "Circuito tras añadir puerta fusionada:" << std::endl;
    std::cout << circuit << std::endl;

    circuit.add_RX_gate(1, 0.5); // añade una rotación RX al qubit 1

    std::cout << "Circuito final:" << std::endl;
    std::cout << circuit << std::endl;

    circuit.update_quantum_state(&state); // aplica el circuito al estado cuántico

    std::cout << "Estado después de aplicar el circuito:" << std::endl;
    std::cout << state << std::endl;

    Observable observable(3); // define un observable hermítico en 3 qubits
    observable.add_operator(2.0, "X 2 Y 1 Z 0");
    observable.add_operator(-3.0, "Z 2");


    auto value = observable.get_expectation_value(&state); // valor esperado del observable

    std::cout << "Valor esperado del observable:" << std::endl;
    std::cout << value << std::endl;

    return 0;
}
