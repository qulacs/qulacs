
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "constant.h"
#include "update_ops.h"
#include "utility.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void single_qubit_diagonal_matrix_gate(UINT target_qubit_index, const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {

	// loop variables
	const ITYPE loop_dim = dim;
	ITYPE state_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// determin matrix pos
		UINT bit_val = (state_index >> target_qubit_index) % 2;

		// set value
		state[state_index] *= diagonal_matrix[bit_val];
	}
}

void single_qubit_phase_gate(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {

	// target tmask
	const ITYPE mask = 1ULL << target_qubit_index;

	// loop varaibles
	const ITYPE loop_dim = dim / 2;
	ITYPE state_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; ++state_index) {

		// crate index
		ITYPE basis_1 = insert_zero_to_basis_index(state_index, mask, target_qubit_index) ^ mask;

		// set values
		state[basis_1] *= phase;
	}
}