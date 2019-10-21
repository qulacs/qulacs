
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

void P0_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE low_mask = mask - 1;
	const ITYPE high_mask = ~low_mask;

	ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0 ; state_index<loop_dim ; ++state_index){
		ITYPE temp_index = (state_index&low_mask) + ((state_index&high_mask) << 1) + mask;
        state[temp_index] = 0;
    }
}

void P1_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim){
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE low_mask = mask - 1;
	const ITYPE high_mask = ~low_mask;

	ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE temp_index = (state_index&low_mask) + ((state_index&high_mask) << 1);
		state[temp_index] = 0;
	}
}
