
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


//void single_qubit_control_single_qubit_dense_matrix_gate_single(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);
//void single_qubit_control_single_qubit_dense_matrix_gate_old_single(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);
//void single_qubit_control_single_qubit_dense_matrix_gate_old_parallel(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);


void single_qubit_control_single_qubit_dense_matrix_gate(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	//single_qubit_control_single_qubit_dense_matrix_gate_old_single(control_qubit_index, control_value, target_qubit_index,matrix,state, dim);
	//single_qubit_control_single_qubit_dense_matrix_gate_old_parallel(control_qubit_index, control_value, target_qubit_index, matrix, state, dim);
	//single_qubit_control_single_qubit_dense_matrix_gate_single(control_qubit_index, control_value, target_qubit_index, matrix, state, dim);
	//single_qubit_control_single_qubit_dense_matrix_gate_single_unroll(control_qubit_index, control_value, target_qubit_index, matrix, state, dim);
	//single_qubit_control_single_qubit_dense_matrix_gate_single_simd(control_qubit_index, control_value, target_qubit_index, matrix, state, dim);
	//single_qubit_control_single_qubit_dense_matrix_gate_parallel_simd(control_qubit_index, control_value, target_qubit_index, matrix, state, dim);

#ifdef _USE_SIMD
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		single_qubit_control_single_qubit_dense_matrix_gate_single_simd(control_qubit_index, control_value, target_qubit_index, matrix, state, dim);
	}
	else {
		single_qubit_control_single_qubit_dense_matrix_gate_parallel_simd(control_qubit_index, control_value, target_qubit_index, matrix, state, dim);
	}
#else
	single_qubit_control_single_qubit_dense_matrix_gate_single_simd(control_qubit_index, control_value, target_qubit_index, matrix, state, dim);
#endif
#else
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		single_qubit_control_single_qubit_dense_matrix_gate_single_unroll(control_qubit_index, control_value, target_qubit_index, matrix, state, dim);
	}
	else {
		single_qubit_control_single_qubit_dense_matrix_gate_parallel_unroll(control_qubit_index, control_value, target_qubit_index, matrix, state, dim);
	}
#else
	single_qubit_control_single_qubit_dense_matrix_gate_single_unroll(control_qubit_index, control_value, target_qubit_index, matrix, state, dim);
#endif
#endif
}

void single_qubit_control_single_qubit_dense_matrix_gate_single_unroll(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index;
	if (target_qubit_index == 0) {
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask * control_value;

			// fetch values
			CTYPE cval0 = state[basis_index];
			CTYPE cval1 = state[basis_index+1];

			// set values
			state[basis_index] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_index+1] = matrix[2] * cval0 + matrix[3] * cval1;
		}
	}
	else if (control_qubit_index == 0) {
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask * control_value;
			ITYPE basis_index_1 = basis_index_0 + target_mask;

			// fetch values
			CTYPE cval0 = state[basis_index_0];
			CTYPE cval1 = state[basis_index_1];

			// set values
			state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
		}
	}else {
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask * control_value;
			ITYPE basis_index_1 = basis_index_0 + target_mask;

			// fetch values
			CTYPE cval0 = state[basis_index_0];
			CTYPE cval1 = state[basis_index_1];
			CTYPE cval2 = state[basis_index_0 + 1];
			CTYPE cval3 = state[basis_index_1 + 1];

			// set values
			state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
			state[basis_index_0 + 1] = matrix[0] * cval2 + matrix[1] * cval3;
			state[basis_index_1 + 1] = matrix[2] * cval2 + matrix[3] * cval3;
		}
	}
}

#ifdef _OPENMP
void single_qubit_control_single_qubit_dense_matrix_gate_parallel_unroll(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index;
	if (target_qubit_index == 0) {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask * control_value;

			// fetch values
			CTYPE cval0 = state[basis_index];
			CTYPE cval1 = state[basis_index + 1];

			// set values
			state[basis_index] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_index + 1] = matrix[2] * cval0 + matrix[3] * cval1;
		}
	}
	else if (control_qubit_index == 0) {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask * control_value;
			ITYPE basis_index_1 = basis_index_0 + target_mask;

			// fetch values
			CTYPE cval0 = state[basis_index_0];
			CTYPE cval1 = state[basis_index_1];

			// set values
			state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
		}
	}
	else {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask * control_value;
			ITYPE basis_index_1 = basis_index_0 + target_mask;

			// fetch values
			CTYPE cval0 = state[basis_index_0];
			CTYPE cval1 = state[basis_index_1];
			CTYPE cval2 = state[basis_index_0 + 1];
			CTYPE cval3 = state[basis_index_1 + 1];

			// set values
			state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
			state[basis_index_0 + 1] = matrix[0] * cval2 + matrix[1] * cval3;
			state[basis_index_1 + 1] = matrix[2] * cval2 + matrix[3] * cval3;
		}
	}
}
#endif

#ifdef _USE_SIMD
void single_qubit_control_single_qubit_dense_matrix_gate_single_simd(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index;
	if (target_qubit_index == 0) {
		__m256d mv00 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]), -cimag(matrix[0]), creal(matrix[0]));
		__m256d mv01 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]), creal(matrix[0]), cimag(matrix[0]));
		__m256d mv20 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]), -cimag(matrix[2]), creal(matrix[2]));
		__m256d mv21 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]), creal(matrix[2]), cimag(matrix[2]));
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask * control_value;
			double* ptr = (double*)(state + basis);
			__m256d data = _mm256_loadu_pd(ptr);

			__m256d data_u0 = _mm256_mul_pd(data, mv00);
			__m256d data_u1 = _mm256_mul_pd(data, mv01);
			__m256d data_u2 = _mm256_hadd_pd(data_u0, data_u1);
			data_u2 = _mm256_permute4x64_pd(data_u2, 216); // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

			__m256d data_d0 = _mm256_mul_pd(data, mv20);
			__m256d data_d1 = _mm256_mul_pd(data, mv21);
			__m256d data_d2 = _mm256_hadd_pd(data_d0, data_d1);
			data_d2 = _mm256_permute4x64_pd(data_d2, 216); // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

			__m256d data_r = _mm256_hadd_pd(data_u2, data_d2);

			data_r = _mm256_permute4x64_pd(data_r, 216); // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216
			_mm256_storeu_pd(ptr, data_r);
		}
	}
	else if (control_qubit_index == 0) {
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask * control_value;
			ITYPE basis_index_1 = basis_index_0 + target_mask;

			// fetch values
			CTYPE cval0 = state[basis_index_0];
			CTYPE cval1 = state[basis_index_1];

			// set values
			state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
		}
	}
	else {
		__m256d mv00 = _mm256_set_pd(-cimag(matrix[0]), creal(matrix[0]), -cimag(matrix[0]), creal(matrix[0]));
		__m256d mv01 = _mm256_set_pd(creal(matrix[0]), cimag(matrix[0]), creal(matrix[0]), cimag(matrix[0]));
		__m256d mv10 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]), -cimag(matrix[1]), creal(matrix[1]));
		__m256d mv11 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]), creal(matrix[1]), cimag(matrix[1]));
		__m256d mv20 = _mm256_set_pd(-cimag(matrix[2]), creal(matrix[2]), -cimag(matrix[2]), creal(matrix[2]));
		__m256d mv21 = _mm256_set_pd(creal(matrix[2]), cimag(matrix[2]), creal(matrix[2]), cimag(matrix[2]));
		__m256d mv30 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]), -cimag(matrix[3]), creal(matrix[3]));
		__m256d mv31 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]), creal(matrix[3]), cimag(matrix[3]));
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask * control_value;
			ITYPE basis_1 = basis_0 + target_mask;

			double* ptr0 = (double*)(state + basis_0);
			double* ptr1 = (double*)(state + basis_1);
			__m256d data0 = _mm256_loadu_pd(ptr0);
			__m256d data1 = _mm256_loadu_pd(ptr1);

			__m256d data_u2 = _mm256_mul_pd(data0, mv00);
			__m256d data_u3 = _mm256_mul_pd(data1, mv10);
			__m256d data_u4 = _mm256_mul_pd(data0, mv01);
			__m256d data_u5 = _mm256_mul_pd(data1, mv11);

			__m256d data_u6 = _mm256_hadd_pd(data_u2, data_u4);
			__m256d data_u7 = _mm256_hadd_pd(data_u3, data_u5);

			__m256d data_d2 = _mm256_mul_pd(data0, mv20);
			__m256d data_d3 = _mm256_mul_pd(data1, mv30);
			__m256d data_d4 = _mm256_mul_pd(data0, mv21);
			__m256d data_d5 = _mm256_mul_pd(data1, mv31);

			__m256d data_d6 = _mm256_hadd_pd(data_d2, data_d4);
			__m256d data_d7 = _mm256_hadd_pd(data_d3, data_d5);

			__m256d data_r0 = _mm256_add_pd(data_u6, data_u7);
			__m256d data_r1 = _mm256_add_pd(data_d6, data_d7);

			_mm256_storeu_pd(ptr0, data_r0);
			_mm256_storeu_pd(ptr1, data_r1);
		}
	}
}

#ifdef _OPENMP
void single_qubit_control_single_qubit_dense_matrix_gate_parallel_simd(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index;
	if (target_qubit_index == 0) {
		__m256d mv00 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]), -cimag(matrix[0]), creal(matrix[0]));
		__m256d mv01 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]), creal(matrix[0]), cimag(matrix[0]));
		__m256d mv20 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]), -cimag(matrix[2]), creal(matrix[2]));
		__m256d mv21 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]), creal(matrix[2]), cimag(matrix[2]));
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask * control_value;
			double* ptr = (double*)(state + basis);
			__m256d data = _mm256_loadu_pd(ptr);

			__m256d data_u0 = _mm256_mul_pd(data, mv00);
			__m256d data_u1 = _mm256_mul_pd(data, mv01);
			__m256d data_u2 = _mm256_hadd_pd(data_u0, data_u1);
			data_u2 = _mm256_permute4x64_pd(data_u2, 216); // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

			__m256d data_d0 = _mm256_mul_pd(data, mv20);
			__m256d data_d1 = _mm256_mul_pd(data, mv21);
			__m256d data_d2 = _mm256_hadd_pd(data_d0, data_d1);
			data_d2 = _mm256_permute4x64_pd(data_d2, 216); // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

			__m256d data_r = _mm256_hadd_pd(data_u2, data_d2);

			data_r = _mm256_permute4x64_pd(data_r, 216); // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216
			_mm256_storeu_pd(ptr, data_r);
		}
	}
	else if (control_qubit_index == 0) {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask * control_value;
			ITYPE basis_index_1 = basis_index_0 + target_mask;

			// fetch values
			CTYPE cval0 = state[basis_index_0];
			CTYPE cval1 = state[basis_index_1];

			// set values
			state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
		}
	}
	else {
		__m256d mv00 = _mm256_set_pd(-cimag(matrix[0]), creal(matrix[0]), -cimag(matrix[0]), creal(matrix[0]));
		__m256d mv01 = _mm256_set_pd(creal(matrix[0]), cimag(matrix[0]), creal(matrix[0]), cimag(matrix[0]));
		__m256d mv10 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]), -cimag(matrix[1]), creal(matrix[1]));
		__m256d mv11 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]), creal(matrix[1]), cimag(matrix[1]));
		__m256d mv20 = _mm256_set_pd(-cimag(matrix[2]), creal(matrix[2]), -cimag(matrix[2]), creal(matrix[2]));
		__m256d mv21 = _mm256_set_pd(creal(matrix[2]), cimag(matrix[2]), creal(matrix[2]), cimag(matrix[2]));
		__m256d mv30 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]), -cimag(matrix[3]), creal(matrix[3]));
		__m256d mv31 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]), creal(matrix[3]), cimag(matrix[3]));
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask * control_value;
			ITYPE basis_1 = basis_0 + target_mask;

			double* ptr0 = (double*)(state + basis_0);
			double* ptr1 = (double*)(state + basis_1);
			__m256d data0 = _mm256_loadu_pd(ptr0);
			__m256d data1 = _mm256_loadu_pd(ptr1);

			__m256d data_u2 = _mm256_mul_pd(data0, mv00);
			__m256d data_u3 = _mm256_mul_pd(data1, mv10);
			__m256d data_u4 = _mm256_mul_pd(data0, mv01);
			__m256d data_u5 = _mm256_mul_pd(data1, mv11);

			__m256d data_u6 = _mm256_hadd_pd(data_u2, data_u4);
			__m256d data_u7 = _mm256_hadd_pd(data_u3, data_u5);

			__m256d data_d2 = _mm256_mul_pd(data0, mv20);
			__m256d data_d3 = _mm256_mul_pd(data1, mv30);
			__m256d data_d4 = _mm256_mul_pd(data0, mv21);
			__m256d data_d5 = _mm256_mul_pd(data1, mv31);

			__m256d data_d6 = _mm256_hadd_pd(data_d2, data_d4);
			__m256d data_d7 = _mm256_hadd_pd(data_d3, data_d5);

			__m256d data_r0 = _mm256_add_pd(data_u6, data_u7);
			__m256d data_r1 = _mm256_add_pd(data_d6, data_d7);

			_mm256_storeu_pd(ptr0, data_r0);
			_mm256_storeu_pd(ptr1, data_r1);
		}
	}
}
#endif
#endif




/*
void single_qubit_control_single_qubit_dense_matrix_gate_old_single(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	// loop varaibles
	const ITYPE loop_dim = dim >> 2;
	ITYPE state_index;
	// mask
	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = (1ULL << control_qubit_index) * control_value;
	// insert index
	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << max_qubit_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create base index
		ITYPE basis_c_t0 = state_index;
		basis_c_t0 = insert_zero_to_basis_index(basis_c_t0, min_qubit_mask, min_qubit_index);
		basis_c_t0 = insert_zero_to_basis_index(basis_c_t0, max_qubit_mask, max_qubit_index);

		// flip control
		basis_c_t0 ^= control_mask;

		// gather index
		ITYPE basis_c_t1 = basis_c_t0 ^ target_mask;

		// fetch values
		CTYPE cval_c_t0 = state[basis_c_t0];
		CTYPE cval_c_t1 = state[basis_c_t1];

		// set values
		state[basis_c_t0] = matrix[0] * cval_c_t0 + matrix[1] * cval_c_t1;
		state[basis_c_t1] = matrix[2] * cval_c_t0 + matrix[3] * cval_c_t1;
	}
}

#ifdef _OPENMP
void single_qubit_control_single_qubit_dense_matrix_gate_old_parallel(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	// loop varaibles
	const ITYPE loop_dim = dim >> 2;
	ITYPE state_index;
	// mask
	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = (1ULL << control_qubit_index) * control_value;
	// insert index
	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << max_qubit_index;

#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create base index
		ITYPE basis_c_t0 = state_index;
		basis_c_t0 = insert_zero_to_basis_index(basis_c_t0, min_qubit_mask, min_qubit_index);
		basis_c_t0 = insert_zero_to_basis_index(basis_c_t0, max_qubit_mask, max_qubit_index);

		// flip control
		basis_c_t0 ^= control_mask;

		// gather index
		ITYPE basis_c_t1 = basis_c_t0 ^ target_mask;

		// fetch values
		CTYPE cval_c_t0 = state[basis_c_t0];
		CTYPE cval_c_t1 = state[basis_c_t1];

		// set values
		state[basis_c_t0] = matrix[0] * cval_c_t0 + matrix[1] * cval_c_t1;
		state[basis_c_t1] = matrix[2] * cval_c_t0 + matrix[3] * cval_c_t1;
	}
}
#endif

void single_qubit_control_single_qubit_dense_matrix_gate_single(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&low_mask)
			+ ((state_index&mid_mask) << 1)
			+ ((state_index&high_mask) << 2)
			+ control_mask * control_value;
		ITYPE basis_index_1 = basis_index_0 + target_mask;

		// fetch values
		CTYPE cval0 = state[basis_index_0];
		CTYPE cval1 = state[basis_index_1];

		// set values
		state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
		state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
	}
}

*/