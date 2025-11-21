#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

// Incluye tu archivo donde ya está definida:
// - ECR_gate_parallel_unroll
// - CTYPE, UINT, ITYPE
// - etc.
#include "csim/update_ops.hpp"

int main() {
    using std::cout;
    using std::endl;

    // ============================================================
    // CONFIGURACIÓN DEL TEST
    // ============================================================
    const UINT q0 = 0;
    const UINT q1 = 4;
    const UINT n_qubits = 5;
    const ITYPE dim = 1ULL << n_qubits;

    cout << "Probando ECR_gate_parallel_unroll sobre " << n_qubits 
         << " qubits..." << endl;

    // ============================================================
    // CREAMOS UN ESTADO SENCILLO
    // ============================================================


    std::vector<CTYPE> state(dim, CTYPE(0.0, 0.0));
    state[0] = CTYPE(0.0101105955, 0.0167271371);
    state[1] = CTYPE(0.1585987468, 0.0808763846);
    state[2] = CTYPE(0.2820801376, 0.0480186846);
    state[3] = CTYPE(-0.0240327857, -0.0574569032);
    state[4] = CTYPE(0.0858650333, 0.2115784928);
    state[5] = CTYPE(0.1657284343, -0.0831781034);
    state[6] = CTYPE(-0.1367750715, -0.1365907697);
    state[7] = CTYPE(-0.0882971706, 0.0984543525);
    state[8] = CTYPE(0.1204164159, -0.0667672630);
    state[9] = CTYPE(0.0914460776, -0.1067782961);
    state[10] = CTYPE(0.0363543846, 0.0023833030);
    state[11] = CTYPE(-0.0767846302, 0.1234282081);
    state[12] = CTYPE(0.0066363518, -0.0148286465);
    state[13] = CTYPE(0.0129140401, -0.0271140316);
    state[14] = CTYPE(-0.0225181255, 0.2397009974);
    state[15] = CTYPE(-0.0890724897, 0.1518390685);
    state[16] = CTYPE(0.1228845220, 0.0055693079);
    state[17] = CTYPE(0.1737948010, 0.1825285070);
    state[18] = CTYPE(-0.0788327840, -0.0068239610);
    state[19] = CTYPE(0.1435911529, 0.1538036339);
    state[20] = CTYPE(-0.0543893547, -0.1873199665);
    state[21] = CTYPE(0.1588434092, -0.0838706501);
    state[22] = CTYPE(-0.0560615395, -0.0229029912);
    state[23] = CTYPE(-0.2549819852, 0.0476732944);
    state[24] = CTYPE(0.0513528643, -0.2798395514);
    state[25] = CTYPE(-0.1730591396, 0.0146931346);
    state[26] = CTYPE(0.0491209904, -0.0320089085);
    state[27] = CTYPE(0.2103362291, -0.0238378796);
    state[28] = CTYPE(0.0636509085, -0.0305138954);
    state[29] = CTYPE(-0.0996959250, -0.3243868490);
    state[30] = CTYPE(-0.1503251987, 0.1018871130);
    state[31] = CTYPE(0.0717523446, -0.0374508447);


    // ============================================================
    // APLICAMOS TU PUERTA ECR OPTIMIZADA
    // ============================================================
    ECR_gate_parallel_simd(q0, q1, state.data(), dim);

    cout << "\nEstado después de aplicar ECR_gate_parallel_unroll:\n";
    for (ITYPE i = 0; i < dim; i++) {
        cout << i << ": (" << std::real(state[i])
             << ", " << std::imag(state[i]) << ")\n";
    } 

    return 0;
}
