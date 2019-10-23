
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "constant.h"
#include "utility.h"
#include "update_ops.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

//void double_qubit_dense_matrix_gate_old_single(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim);
//void double_qubit_dense_matrix_gate_old_parallel(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim);

void double_qubit_dense_matrix_gate_c(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim) {
	//double_qubit_dense_matrix_gate_old_single(target_qubit_index1, target_qubit_index2, matrix,state, dim);
	//double_qubit_dense_matrix_gate_old_parallel(target_qubit_index1, target_qubit_index2, matrix, state, dim);
	//double_qubit_dense_matrix_gate_single(target_qubit_index1, target_qubit_index2, matrix, state, dim);
	//double_qubit_dense_matrix_gate_parallel(target_qubit_index1, target_qubit_index2, matrix, state, dim);

#ifdef _OPENMP
	UINT threshold = 9;
	if (dim < (((ITYPE)1) << threshold)) {
		double_qubit_dense_matrix_gate_single(target_qubit_index1, target_qubit_index2, matrix, state, dim);
	}
	else {
		double_qubit_dense_matrix_gate_parallel(target_qubit_index1, target_qubit_index2, matrix, state, dim);
	}
#else
	double_qubit_dense_matrix_gate_single(target_qubit_index1, target_qubit_index2, matrix, state, dim);
#endif
}

void double_qubit_dense_matrix_gate_single(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim) {
	const UINT min_qubit_index = get_min_ui(target_qubit_index1, target_qubit_index2);
	const UINT max_qubit_index = get_max_ui(target_qubit_index1, target_qubit_index2);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	const ITYPE target_mask1 = 1ULL << target_qubit_index1;
	const ITYPE target_mask2 = 1ULL << target_qubit_index2;

	// loop variables
	const ITYPE loop_dim = dim / 4;
	ITYPE state_index;

	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create index
		ITYPE basis_0 = (state_index&low_mask)
			+ ((state_index&mid_mask) << 1)
			+ ((state_index&high_mask) << 2);

		// gather index
		ITYPE basis_1 = basis_0 + target_mask1;
		ITYPE basis_2 = basis_0 + target_mask2;
		ITYPE basis_3 = basis_1 + target_mask2;

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

#ifdef _OPENMP
void double_qubit_dense_matrix_gate_parallel(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim) {
	const UINT min_qubit_index = get_min_ui(target_qubit_index1, target_qubit_index2);
	const UINT max_qubit_index = get_max_ui(target_qubit_index1, target_qubit_index2);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	const ITYPE target_mask1 = 1ULL << target_qubit_index1;
	const ITYPE target_mask2 = 1ULL << target_qubit_index2;

	// loop variables
	const ITYPE loop_dim = dim / 4;
	ITYPE state_index;

#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create index
		ITYPE basis_0 = (state_index&low_mask)
			+ ((state_index&mid_mask) << 1)
			+ ((state_index&high_mask) << 2);

		// gather index
		ITYPE basis_1 = basis_0 + target_mask1;
		ITYPE basis_2 = basis_0 + target_mask2;
		ITYPE basis_3 = basis_1 + target_mask2;

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
#endif



/*
void double_qubit_dense_matrix_gate_old_single(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim) {

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

#ifdef _OPENMP
void double_qubit_dense_matrix_gate_old_parallel(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim) {

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

#pragma omp parallel for
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
#endif
*/