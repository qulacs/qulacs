#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
//#undef _USE_SVE
#include "csim/update_ops.hpp"
#include <cppsim/state.hpp>

using CTYPE = std::complex<double>;
using UINT  = unsigned int;
using ITYPE = unsigned long long;

// tu función de impresión (la original)
void print_state(const CTYPE* state, ITYPE dim) {
    for (ITYPE i = 0; i < dim; ++i) {
        std::cout << state[i] << "\n";
    }
}

int main() {
    const UINT nqubits = 5;
    // construye el estado en la base |0001> (según tu clase)
    QuantumState state(nqubits, 1);
    state.set_Haar_random_state(2023);

    // 1) accede a dim como miembro (sin paréntesis) si es un campo
    ITYPE dim = state.dim; // <-- sin '()'

    // 2) state.data() devuelve void*, por tanto hay que hacer cast
    const CTYPE* state_const_ptr = reinterpret_cast<const CTYPE*>(state.data());
    CTYPE*       state_ptr       = reinterpret_cast<CTYPE*>(state.data());

    std::cout << "Estado inicial:\n";
    print_state(state_const_ptr, dim);

    // ====== Aplicar puerta ECR ======
    std::cout << "\nAplicando SWAP_gate(0,2)...\n";
    SWAP_gate(0, 2, state_ptr, dim);

    std::cout << "\nEstado final:\n";
    print_state(state_const_ptr, dim);

    std::cout << "\nEsperado (aproximado): 0.7071|01> + 0.7071i|11>\n";
    return 0;
}
