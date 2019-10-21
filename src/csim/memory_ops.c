#include <stdio.h>
#include <stdlib.h>
#include "memory_ops.h"
#include "utility.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#define aligned_free _aligned_free;
#define aligned_malloc _aligned_malloc;
#endif

// memory allocation
CTYPE* allocate_quantum_state(ITYPE dim) {
    CTYPE* state = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*dim));
	//CTYPE* state = (CTYPE*)_aligned_malloc((size_t)(sizeof(CTYPE)*dim), 32);

    if (!state){
        fprintf(stderr,"Out of memory\n");
		fflush(stderr);
        exit(1);
    }
    return state;
}

void release_quantum_state(CTYPE* state) {
	free(state);
	//_aligned_free(state);
}

