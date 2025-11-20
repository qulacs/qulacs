#include <iostream>

#include "cppsim/circuit.hpp"

// Isto é unha proba cambiando a liña UINT _qubit_count; de circuit.hpp de protected a public para poder acceder a ela.

int main() {

    QuantumCircuit qc_uxia(5);

    std::cout << "El número de qubits es: " << qc_uxia._qubit_count << std::endl;

    std::cout << "Esta é a miña primeira proba de execución" << "\n";

    return 0;
}
