

#include <mpi.h>

#include <cmath>
#include <complex>
#include <cppsim/state.hpp>
#include <iostream>
#include <vector>

#include "csim/update_ops.hpp"

// Simula typedefs del código original
using UINT = unsigned int;
using ITYPE = unsigned long long;
using CTYPE = std::complex<double>;

// Definición (puede ir después del main o antes)
/*void print_state(const CTYPE* state, size_t n) {
    std::cout << "Inside print_state\n";
    for (size_t i = 0; i < n; ++i) {
        std::cout << " state[" << i << "] : ("
                  << state[i].real() << ", " << state[i].imag() << ")\n";
    }
}
*/

void print_state_local(
    const CTYPE* local_state, UINT inner_qc, int mpirank, int mpisize) {
    ITYPE local_dim = 1ULL << inner_qc;
    ITYPE offset =
        (ITYPE)mpirank * local_dim;  // índice global inicial para este rank

    // Sincronizamos salidas para que no se entremezclen demasiado
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "=== rank " << mpirank << " (offset " << offset
              << ", local_dim " << local_dim << ") ===\n";
    for (ITYPE i = 0; i < local_dim; ++i) {
        ITYPE global_index = offset + i;
        std::cout << " state[" << global_index << "] = ("
                  << local_state[i].real() << ", " << local_state[i].imag()
                  << ")\n";
    }
    std::cout << std::flush;
    // Asegura orden de impresión: siguiente rank sale tras barrier local
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_state_gather(
    const CTYPE* local_state, UINT inner_qc, int mpirank, int mpisize) {
    ITYPE local_dim = 1ULL << inner_qc;
    ITYPE global_dim = local_dim * (ITYPE)mpisize;

    // Buffer sólo en rank 0
    std::vector<CTYPE> full;
    if (mpirank == 0) full.resize((size_t)global_dim);

    // Reunimos como bytes para evitar problemas con tipos MPI para std::complex
    MPI_Gather((void*)local_state, (int)(local_dim * sizeof(CTYPE)), MPI_BYTE,
        mpirank == 0 ? (void*)full.data() : nullptr,
        (int)(local_dim * sizeof(CTYPE)), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (mpirank == 0) {
        std::cout << "=== Full state on rank 0 (global_dim = " << global_dim
                  << ") ===\n";
        for (ITYPE i = 0; i < global_dim; ++i) {
            std::cout << " state[" << i << "] = (" << full[(size_t)i].real()
                      << ", " << full[(size_t)i].imag() << ")\n";
        }
        std::cout << std::flush;
    }
}

int main(int argc, char** argv) {
    int mpirank, mpisize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    const UINT p = (UINT)std::log2(mpisize);
    if (mpirank == 0) {
        std::cout << "p: " << p << "\n";
    }

    UINT nqubits = 4;

    if (nqubits >= (p + 2)) {
        std::cout
            << "WARNING: this will not work well with MPI. Use less processes."
            << "\n";
    }

    QuantumState state(nqubits, 1);  // use multi_cpu if possible

    const UINT inner_qc = nqubits - p;

    std::cout << "inner_qc: " << inner_qc << "\n";
    // print_state_local(state._state_vector, inner_qc, mpirank, mpisize);

    const ITYPE dim = 1ULL << inner_qc;

    std::cout << "Dimensión: " << dim << "\n";

    /* const double sqrt2inv = 1./sqrt(2.);
    std::vector<CTYPE> state(dim, 0.);
    state[0] = sqrt2inv;
    state[3] = sqrt2inv; */

    std::cout << "Estado inicial:\n";
    // print_state(state.data());
    print_state_gather(state._state_vector, inner_qc, mpirank, mpisize);

    // ====== Aplicar puerta ECR ======
    std::cout << "\nAplicando ECR_gate(1,2)...\n";

    /* if (mpirank == 0)
        print_state(state._state_vector); */
    // print_state_local(state._state_vector, inner_qc, mpirank, mpisize);
    //  print_state_gather(state._state_vector, inner_qc, mpirank, mpisize);
    ECR_gate_mpi(1, 2, state._state_vector, dim, inner_qc);
    // print_state_local(state._state_vector, inner_qc, mpirank, mpisize);
    print_state_gather(state._state_vector, inner_qc, mpirank, mpisize);

    /* std::cout << "\nEstado final no rango:" << rank << "\n";
    print_state(state.data()); */

    std::cout << "\nAcabou\n";

    MPI_Finalize();

    return 0;
}
