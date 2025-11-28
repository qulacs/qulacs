#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include "update_ops.hpp"

// Simula typedefs del código original
using UINT = unsigned int;
using ITYPE = unsigned long long;
using CTYPE = std::complex<double>;

// Declaración
void print_state(const CTYPE* state, ITYPE dim);

// Definición (puede ir después del main o antes)
void print_state(const CTYPE* state, ITYPE dim) {
    for (ITYPE i = 0; i < dim; ++i) {
        std::cout << " |" << i << "> : " << state[i] << std::endl;
    }
}



int main() {
    const UINT nqubits = 4;
    const ITYPE dim = 1ULL << nqubits;

    std::vector<CTYPE> state(dim, 0.);


    std::cout << "Estado inicial:\n";
    print_state(state.data(), dim);

    // ====== Aplicar puerta ECR ======
    std::cout << "\nAplicando ECR_gate(0,1)...\n";
    ECR_gate(0, 1, state.data(), dim);

    std::cout << "\nEstado final:\n";
    print_state(state.data(), dim);


    std::cout << "\nEsperado (aproximado): 0.7071|01> + 0.7071i|11>\n";
}
