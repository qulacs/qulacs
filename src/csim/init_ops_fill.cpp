#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "MPIutil.hpp"
#include "init_ops.hpp"
#include "utility.hpp"

void initialize_quantum_state(CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 10);
#pragma omp parallel for
#endif
    for (ITYPE index = 0; index < dim; ++index) {
        state[index] = 0;
    }
#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif

    state[0] = 1.0;
}

#ifdef _USE_MPI
void initialize_quantum_state_mpi(CTYPE* state, ITYPE dim, UINT outer_qc) {
#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 10);
#pragma omp parallel for
#endif
    for (ITYPE index = 0; index < dim; ++index) {
        state[index] = 0;
    }
#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif

    MPIutil mpiutil = get_mpiutil();
    if (outer_qc == 0 || mpiutil->get_rank() == 0) state[0] = 1.0;
}
#endif
