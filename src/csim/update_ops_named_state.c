
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "constant.h"
#include "update_ops.h"
#include "utility.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

void normalize(double squared_norm, CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim;
    const double normalize_factor = sqrt(1./squared_norm);
    ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0 ; state_index<loop_dim ; ++state_index){
        state[state_index] *= normalize_factor;
    }
}

void state_add(const CTYPE *state_added, CTYPE *state, ITYPE dim) {
	ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (index = 0; index < dim; ++index) {
		state[index] += state_added[index];
	}
}

void state_multiply(CTYPE coef, CTYPE *state, ITYPE dim) {
	ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (index = 0; index < dim; ++index) {
		state[index] *= coef;
	}
}

