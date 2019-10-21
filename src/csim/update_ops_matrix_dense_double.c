
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
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

void two_qubit_dense_matrix_gate(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim) {

	// target mask
	const UINT target_qubit_index_min = (target_qubit_index1 < target_qubit_index2 ? target_qubit_index1 : target_qubit_index2);
	const UINT target_qubit_index_max = (target_qubit_index1 >= target_qubit_index2 ? target_qubit_index1 : target_qubit_index2);
	const ITYPE target_mask_min = 1ULL << target_qubit_index_min;
	const ITYPE target_mask_max = 1ULL << target_qubit_index_max;
	const ITYPE target_mask1 = 1ULL << target_qubit_index1;
	const ITYPE target_mask2 = 1ULL << target_qubit_index2;

	// loop variables
	const ITYPE loop_dim = dim / 4;
	ITYPE state_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create index
		ITYPE basis_0 = state_index;
		basis_0 = insert_zero_to_basis_index(basis_0, target_mask_min, target_qubit_index_min);
		basis_0 = insert_zero_to_basis_index(basis_0, target_mask_max, target_qubit_index_max);

		// gather index
		ITYPE basis_1 = basis_0 ^ target_mask1;
		ITYPE basis_2 = basis_0 ^ target_mask2;
		ITYPE basis_3 = basis_0 ^ target_mask1 ^ target_mask2;

		// fetch values
		CTYPE cval_0 = state[basis_0];
		CTYPE cval_1 = state[basis_1];
		CTYPE cval_2 = state[basis_2];
		CTYPE cval_3 = state[basis_3];

		// set values
		state[basis_0] = matrix[0] * cval_0 + matrix[1] * cval_1 + matrix[2] * cval_2 + matrix[3] * cval_3;
		state[basis_1] = matrix[4] * cval_0 + matrix[5] * cval_1 + matrix[6] * cval_2 + matrix[7] * cval_3;
		state[basis_2] = matrix[8] * cval_0 + matrix[9] * cval_1 + matrix[10] * cval_2 + matrix[11] * cval_3;
		state[basis_3] = matrix[12] * cval_0 + matrix[13] * cval_1 + matrix[14] * cval_2 + matrix[15] * cval_3;
	}
}
