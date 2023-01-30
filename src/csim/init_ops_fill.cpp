#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "init_ops.hpp"
#include "utility.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

// state initialization
void initialize_quantum_state_parallel(CTYPE* state, ITYPE dim);
void initialize_quantum_state(CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 15);
#endif

    initialize_quantum_state_parallel(state, dim);

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void initialize_quantum_state_parallel(CTYPE* state, ITYPE dim) {
    ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (index = 0; index < dim; ++index) {
        state[index] = 0;
    }
    state[0] = 1.0;
}
