
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.hpp"
#include "update_ops.hpp"
#include "utility.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void normalize(double squared_norm, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim;
    const double normalize_factor = sqrt(1. / squared_norm);
    ITYPE state_index;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        state[state_index] *= normalize_factor;
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void normalize_single_thread(double squared_norm, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim;
    const double normalize_factor = sqrt(1. / squared_norm);
    ITYPE state_index;
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        state[state_index] *= normalize_factor;
    }
}

// TODO: support when one of the states is multi-cpu
void state_add(const CTYPE* state_added, CTYPE* state, ITYPE dim) {
    ITYPE index;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#pragma omp parallel for
#endif
    for (index = 0; index < dim; ++index) {
        state[index] += state_added[index];
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

// TODO: support when one of the states is multi-cpu
void state_add_with_coef(
    CTYPE coef, const CTYPE* state_added, CTYPE* state, ITYPE dim) {
    ITYPE index;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#pragma omp parallel for
#endif
    for (index = 0; index < dim; ++index) {
        state[index] += coef * state_added[index];
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

// TODO: support when one of the states is multi-cpu
void state_add_with_coef_single_thread(
    CTYPE coef, const CTYPE* state_added, CTYPE* state, ITYPE dim) {
    ITYPE index;
    for (index = 0; index < dim; ++index) {
        state[index] += coef * state_added[index];
    }
}

void state_multiply(CTYPE coef, CTYPE* state, ITYPE dim) {
    ITYPE index;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#pragma omp parallel for
#endif
    for (index = 0; index < dim; ++index) {
        state[index] *= coef;
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}
