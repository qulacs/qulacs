

#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

/*
int main() {
    QuantumState state(2); 

    std::vector<CPPCTYPE> vec = {0.0, 0.0, 0.0, 1.0};
    state.load(vec.data());

    std::cout << "Estado inicial:" << std::endl;
    std::cout << state << std::endl;

    QuantumCircuit circuit(2); // crea un circuito cuántico vacío de 2 qubits

    circuit.add_ECR_gate(0,1);

    circuit.update_quantum_state(&state); // aplica el circuito al estado cuántico

    std::cout << "Estado después de aplicar el circuito:" << std::endl;
    std::cout << state << std::endl;



    return 0;
}
*/


using CTYPE = std::complex<double>;


int main() {
    const int n = 5; // 2^5 = 32 amplitudes
    const ITYPE dim = 1ULL << n;

    // 1️⃣ Creamos el vector con tus amplitudes
    std::vector<CTYPE> initial_state(dim, CTYPE(0.0, 0.0));

    initial_state[0]  = CTYPE(0.0101105955, 0.0167271371);
    initial_state[1]  = CTYPE(0.1585987468, 0.0808763846);
    initial_state[2]  = CTYPE(0.2820801376, 0.0480186846);
    initial_state[3]  = CTYPE(-0.0240327857, -0.0574569032);
    initial_state[4]  = CTYPE(0.0858650333, 0.2115784928);
    initial_state[5]  = CTYPE(0.1657284343, -0.0831781034);
    initial_state[6]  = CTYPE(-0.1367750715, -0.1365907697);
    initial_state[7]  = CTYPE(-0.0882971706, 0.0984543525);
    initial_state[8]  = CTYPE(0.1204164159, -0.0667672630);
    initial_state[9]  = CTYPE(0.0914460776, -0.1067782961);
    initial_state[10] = CTYPE(0.0363543846, 0.0023833030);
    initial_state[11] = CTYPE(-0.0767846302, 0.1234282081);
    initial_state[12] = CTYPE(0.0066363518, -0.0148286465);
    initial_state[13] = CTYPE(0.0129140401, -0.0271140316);
    initial_state[14] = CTYPE(-0.0225181255, 0.2397009974);
    initial_state[15] = CTYPE(-0.0890724897, 0.1518390685);
    initial_state[16] = CTYPE(0.1228845220, 0.0055693079);
    initial_state[17] = CTYPE(0.1737948010, 0.1825285070);
    initial_state[18] = CTYPE(-0.0788327840, -0.0068239610);
    initial_state[19] = CTYPE(0.1435911529, 0.1538036339);
    initial_state[20] = CTYPE(-0.0543893547, -0.1873199665);
    initial_state[21] = CTYPE(0.1588434092, -0.0838706501);
    initial_state[22] = CTYPE(-0.0560615395, -0.0229029912);
    initial_state[23] = CTYPE(-0.2549819852, 0.0476732944);
    initial_state[24] = CTYPE(0.0513528643, -0.2798395514);
    initial_state[25] = CTYPE(-0.1730591396, 0.0146931346);
    initial_state[26] = CTYPE(0.0491209904, -0.0320089085);
    initial_state[27] = CTYPE(0.2103362291, -0.0238378796);
    initial_state[28] = CTYPE(0.0636509085, -0.0305138954);
    initial_state[29] = CTYPE(-0.0996959250, -0.3243868490);
    initial_state[30] = CTYPE(-0.1503251987, 0.1018871130);
    initial_state[31] = CTYPE(0.0717523446, -0.0374508447); 
 
     

    // 2️⃣ Creamos el estado cuántico con n qubits
    QuantumState state(n);

    // 3️⃣ Cargamos el vector inicial en el QuantumState
    state.load(initial_state.data());

    std::cout << "Estado inicial:" << std::endl;
    std::cout << state << std::endl;

    // 4️⃣ Creamos el circuito
    QuantumCircuit circuit(n); 

    // 5️⃣ Añadimos la puerta ECR
    circuit.add_ECR_gate(0,1);

    // 6️⃣ Aplicamos el circuito
    circuit.update_quantum_state(&state);

    std::cout << "Estado después de aplicar el circuito:" << std::endl;
    std::cout << state << std::endl;

    return 0;
}
