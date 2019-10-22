#include <stdio.h>
#include <stdlib.h>
#include "memory_ops_dm.h"
#include "utility.h"
#include <time.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif

CTYPE* dm_allocate_quantum_state(ITYPE dim) {
    CTYPE* state = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*dim*dim));
    if (!state){
        fprintf(stderr,"Out of memory\n");
        exit(1);
    }
    return state;
}

void dm_initialize_quantum_state(CTYPE *state, ITYPE dim) {
    ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(index=0; index < dim*dim ; ++index){
        state[index]=0;
    }
    state[0] = 1.0;
}

void dm_release_quantum_state(CTYPE* state){
    free(state);
}

void dm_initialize_with_pure_state(CTYPE *state, const CTYPE *pure_state, ITYPE dim) {
	ITYPE ind_y;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (ind_y = 0; ind_y < dim; ++ind_y) {
		ITYPE ind_x;
		for (ind_x = 0; ind_x < dim; ++ind_x) {
			state[ind_y*dim+ind_x] = pure_state[ind_y]*conj(pure_state[ind_x]);
		}
	}
}
