
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


void create_shift_mask_list_from_list_and_value_buf(const UINT* array, UINT count, UINT target, UINT* dst_array, ITYPE* dst_mask);

//void multi_qubit_control_single_qubit_dense_matrix_gate_old_single(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);
//void multi_qubit_control_single_qubit_dense_matrix_gate_old_parallel(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);
//void multi_qubit_control_single_qubit_dense_matrix_gate_single(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);
//void multi_qubit_control_single_qubit_dense_matrix_gate_single_stack(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);
//ITYPE* create_shift_mask_list(const UINT* array, UINT count, UINT target);


void create_shift_mask_list_from_list_and_value_buf(const UINT* array, UINT count, UINT target, UINT* dst_array, ITYPE* dst_mask) {
	UINT size = count + 1;
	memcpy(dst_array, array, sizeof(UINT)*count);
	dst_array[count] = target;
	sort_ui(dst_array, size);
	for (UINT i = 0; i < size; ++i) {
		dst_mask[i] = (1UL << dst_array[i]) - 1;
	}
}

void multi_qubit_control_single_qubit_dense_matrix_gate(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count,
	UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {

	//multi_qubit_control_single_qubit_dense_matrix_gate_old_single(control_qubit_index_list, control_value_list, control_qubit_index_count, target_qubit_index, matrix, state, dim);
	//multi_qubit_control_single_qubit_dense_matrix_gate_old_parallel(control_qubit_index_list, control_value_list, control_qubit_index_count, target_qubit_index, matrix, state, dim);
	//multi_qubit_control_single_qubit_dense_matrix_gate_single(control_qubit_index_list, control_value_list, control_qubit_index_count, target_qubit_index, matrix, state, dim);
	//multi_qubit_control_single_qubit_dense_matrix_gate_single_stack(control_qubit_index_list, control_value_list, control_qubit_index_count,target_qubit_index,matrix,state, dim);
	//multi_qubit_control_single_qubit_dense_matrix_gate_single_unroll(control_qubit_index_list, control_value_list, control_qubit_index_count,target_qubit_index,matrix,state, dim);
	//multi_qubit_control_single_qubit_dense_matrix_gate_single_simd(control_qubit_index_list, control_value_list, control_qubit_index_count,target_qubit_index,matrix,state, dim);
	//multi_qubit_control_single_qubit_dense_matrix_gate_parallel_simd(control_qubit_index_list, control_value_list, control_qubit_index_count,target_qubit_index,matrix,state, dim);

	if (control_qubit_index_count == 1) {
		single_qubit_control_single_qubit_dense_matrix_gate(control_qubit_index_list[0], control_value_list[0], target_qubit_index, matrix, state, dim);
		return;
	}

#ifdef _USE_SIMD
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		multi_qubit_control_single_qubit_dense_matrix_gate_single_simd(control_qubit_index_list, control_value_list, control_qubit_index_count,target_qubit_index,matrix,state, dim);
	}
	else {
		multi_qubit_control_single_qubit_dense_matrix_gate_parallel_simd(control_qubit_index_list, control_value_list, control_qubit_index_count,target_qubit_index,matrix,state, dim);
	}
#else
	multi_qubit_control_single_qubit_dense_matrix_gate_single_simd(control_qubit_index_list, control_value_list, control_qubit_index_count, target_qubit_index, matrix, state, dim);
#endif
#else
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		multi_qubit_control_single_qubit_dense_matrix_gate_single_unroll(control_qubit_index_list, control_value_list, control_qubit_index_count, target_qubit_index, matrix, state, dim);
	}
	else {
		multi_qubit_control_single_qubit_dense_matrix_gate_parallel_unroll(control_qubit_index_list, control_value_list, control_qubit_index_count, target_qubit_index, matrix, state, dim);
	}
#else
	multi_qubit_control_single_qubit_dense_matrix_gate_single_unroll(control_qubit_index_list, control_value_list, control_qubit_index_count, target_qubit_index, matrix, state, dim);
#endif
#endif
}


void multi_qubit_control_single_qubit_dense_matrix_gate_single_unroll(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count,
	UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {

	UINT sort_array[64];
	ITYPE mask_array[64];
	create_shift_mask_list_from_list_and_value_buf(control_qubit_index_list, control_qubit_index_count, target_qubit_index, sort_array, mask_array);
	const ITYPE target_mask = 1ULL << target_qubit_index;
	ITYPE control_mask = create_control_mask(control_qubit_index_list, control_value_list, control_qubit_index_count);

	const UINT insert_index_list_count = control_qubit_index_count + 1;
	const ITYPE loop_dim = dim >> insert_index_list_count;

	if (target_qubit_index == 0) {
		ITYPE state_index;
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			// create base index
			ITYPE basis_0 = state_index;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
			}
			basis_0 += control_mask;

			// fetch values
			CTYPE cval0 = state[basis_0];
			CTYPE cval1 = state[basis_0+1];
			// set values
			state[basis_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_0+1] = matrix[2] * cval0 + matrix[3] * cval1;
		}
	}
	else if (sort_array[0] == 0) {
		ITYPE state_index;
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			// create base index
			ITYPE basis_0 = state_index;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
			}
			basis_0 += control_mask;
			ITYPE basis_1 = basis_0 + target_mask;

			// fetch values
			CTYPE cval0 = state[basis_0];
			CTYPE cval1 = state[basis_1];
			// set values
			state[basis_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_1] = matrix[2] * cval0 + matrix[3] * cval1;
		}

	}
	else {
		ITYPE state_index;
		for (state_index = 0; state_index < loop_dim; state_index+=2) {
			// create base index
			ITYPE basis_0 = state_index;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
			}
			basis_0 += control_mask;
			ITYPE basis_1 = basis_0 + target_mask;

			// fetch values
			CTYPE cval0 = state[basis_0];
			CTYPE cval1 = state[basis_1];
			CTYPE cval2 = state[basis_0+1];
			CTYPE cval3 = state[basis_1+1];
			// set values
			state[basis_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_1] = matrix[2] * cval0 + matrix[3] * cval1;
			state[basis_0+1] = matrix[0] * cval2 + matrix[1] * cval3;
			state[basis_1+1] = matrix[2] * cval2 + matrix[3] * cval3;
		}
	}
}

#ifdef _OPENMP
void multi_qubit_control_single_qubit_dense_matrix_gate_parallel_unroll(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count,
	UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {

	UINT sort_array[64];
	ITYPE mask_array[64];
	create_shift_mask_list_from_list_and_value_buf(control_qubit_index_list, control_qubit_index_count, target_qubit_index, sort_array, mask_array);
	const ITYPE target_mask = 1ULL << target_qubit_index;
	ITYPE control_mask = create_control_mask(control_qubit_index_list, control_value_list, control_qubit_index_count);

	const UINT insert_index_list_count = control_qubit_index_count + 1;
	const ITYPE loop_dim = dim >> insert_index_list_count;

	if (target_qubit_index == 0) {
		ITYPE state_index;
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			// create base index
			ITYPE basis_0 = state_index;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
			}
			basis_0 += control_mask;

			// fetch values
			CTYPE cval0 = state[basis_0];
			CTYPE cval1 = state[basis_0 + 1];
			// set values
			state[basis_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_0 + 1] = matrix[2] * cval0 + matrix[3] * cval1;
		}
	}
	else if (sort_array[0] == 0) {
		ITYPE state_index;
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			// create base index
			ITYPE basis_0 = state_index;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
			}
			basis_0 += control_mask;
			ITYPE basis_1 = basis_0 + target_mask;

			// fetch values
			CTYPE cval0 = state[basis_0];
			CTYPE cval1 = state[basis_1];
			// set values
			state[basis_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_1] = matrix[2] * cval0 + matrix[3] * cval1;
		}

	}
	else {
		ITYPE state_index;
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			// create base index
			ITYPE basis_0 = state_index;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
			}
			basis_0 += control_mask;
			ITYPE basis_1 = basis_0 + target_mask;

			// fetch values
			CTYPE cval0 = state[basis_0];
			CTYPE cval1 = state[basis_1];
			CTYPE cval2 = state[basis_0 + 1];
			CTYPE cval3 = state[basis_1 + 1];
			// set values
			state[basis_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_1] = matrix[2] * cval0 + matrix[3] * cval1;
			state[basis_0 + 1] = matrix[0] * cval2 + matrix[1] * cval3;
			state[basis_1 + 1] = matrix[2] * cval2 + matrix[3] * cval3;
		}
	}
}
#endif



#ifdef _USE_SIMD
void multi_qubit_control_single_qubit_dense_matrix_gate_single_simd(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count,
	UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {

	UINT sort_array[64];
	ITYPE mask_array[64];
	create_shift_mask_list_from_list_and_value_buf(control_qubit_index_list, control_qubit_index_count, target_qubit_index, sort_array, mask_array);
	const ITYPE target_mask = 1ULL << target_qubit_index;
	ITYPE control_mask = create_control_mask(control_qubit_index_list, control_value_list, control_qubit_index_count);

	const UINT insert_index_list_count = control_qubit_index_count + 1;
	const ITYPE loop_dim = dim >> insert_index_list_count;

	if (target_qubit_index == 0) {
		__m256d mv00 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]), -cimag(matrix[0]), creal(matrix[0]));
		__m256d mv01 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]), creal(matrix[0]), cimag(matrix[0]));
		__m256d mv20 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]), -cimag(matrix[2]), creal(matrix[2]));
		__m256d mv21 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]), creal(matrix[2]), cimag(matrix[2]));
		ITYPE state_index;
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			// create base index
			ITYPE basis_0 = state_index;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
			}
			basis_0 += control_mask;

			double* ptr = (double*)(state + basis_0);
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
	else if (sort_array[0] == 0) {
		ITYPE state_index;
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			// create base index
			ITYPE basis_0 = state_index;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
			}
			basis_0 += control_mask;
			ITYPE basis_1 = basis_0 + target_mask;

			// fetch values
			CTYPE cval0 = state[basis_0];
			CTYPE cval1 = state[basis_1];
			// set values
			state[basis_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_1] = matrix[2] * cval0 + matrix[3] * cval1;
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
		ITYPE state_index;
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			// create base index
			ITYPE basis_0 = state_index;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
			}
			basis_0 += control_mask;
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
void multi_qubit_control_single_qubit_dense_matrix_gate_parallel_simd(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count,
	UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {

	UINT sort_array[64];
	ITYPE mask_array[64];
	create_shift_mask_list_from_list_and_value_buf(control_qubit_index_list, control_qubit_index_count, target_qubit_index, sort_array, mask_array);
	const ITYPE target_mask = 1ULL << target_qubit_index;
	ITYPE control_mask = create_control_mask(control_qubit_index_list, control_value_list, control_qubit_index_count);

	const UINT insert_index_list_count = control_qubit_index_count + 1;
	const ITYPE loop_dim = dim >> insert_index_list_count;

	if (target_qubit_index == 0) {
		__m256d mv00 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]), -cimag(matrix[0]), creal(matrix[0]));
		__m256d mv01 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]), creal(matrix[0]), cimag(matrix[0]));
		__m256d mv20 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]), -cimag(matrix[2]), creal(matrix[2]));
		__m256d mv21 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]), creal(matrix[2]), cimag(matrix[2]));
		ITYPE state_index;
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			// create base index
			ITYPE basis_0 = state_index;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
			}
			basis_0 += control_mask;

			double* ptr = (double*)(state + basis_0);
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
	else if (sort_array[0] == 0) {
		ITYPE state_index;
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			// create base index
			ITYPE basis_0 = state_index;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
			}
			basis_0 += control_mask;
			ITYPE basis_1 = basis_0 + target_mask;

			// fetch values
			CTYPE cval0 = state[basis_0];
			CTYPE cval1 = state[basis_1];
			// set values
			state[basis_0] = matrix[0] * cval0 + matrix[1] * cval1;
			state[basis_1] = matrix[2] * cval0 + matrix[3] * cval1;
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
		ITYPE state_index;
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			// create base index
			ITYPE basis_0 = state_index;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
			}
			basis_0 += control_mask;
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
void multi_qubit_control_single_qubit_dense_matrix_gate_old_single(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count,
	UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	// insert index list
	const UINT insert_index_list_count = control_qubit_index_count + 1;
	UINT* insert_index_list = create_sorted_ui_list_value(control_qubit_index_list, control_qubit_index_count, target_qubit_index);
	// target mask
	const ITYPE target_mask = 1ULL << target_qubit_index;
	// control mask
	ITYPE control_mask = create_control_mask(control_qubit_index_list, control_value_list, control_qubit_index_count);
	// loop variables
	const ITYPE loop_dim = dim >> insert_index_list_count;
	ITYPE state_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create base index
		ITYPE basis_c_t0 = state_index;
		for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
			basis_c_t0 = insert_zero_to_basis_index(basis_c_t0, 1ULL << insert_index_list[cursor], insert_index_list[cursor]);
		}
		// flip controls
		basis_c_t0 ^= control_mask;
		// gather target
		ITYPE basis_c_t1 = basis_c_t0 ^ target_mask;
		// fetch values
		CTYPE cval_c_t0 = state[basis_c_t0];
		CTYPE cval_c_t1 = state[basis_c_t1];
		// set values
		state[basis_c_t0] = matrix[0] * cval_c_t0 + matrix[1] * cval_c_t1;
		state[basis_c_t1] = matrix[2] * cval_c_t0 + matrix[3] * cval_c_t1;
	}

	free(insert_index_list);
}

#ifdef _OPENMP
void multi_qubit_control_single_qubit_dense_matrix_gate_old_parallel(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count,
	UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	// insert index list
	const UINT insert_index_list_count = control_qubit_index_count + 1;
	UINT* insert_index_list = create_sorted_ui_list_value(control_qubit_index_list, control_qubit_index_count, target_qubit_index);
	// target mask
	const ITYPE target_mask = 1ULL << target_qubit_index;
	// control mask
	ITYPE control_mask = create_control_mask(control_qubit_index_list, control_value_list, control_qubit_index_count);
	// loop variables
	const ITYPE loop_dim = dim >> insert_index_list_count;
	ITYPE state_index;
#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create base index
		ITYPE basis_c_t0 = state_index;
		for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
			basis_c_t0 = insert_zero_to_basis_index(basis_c_t0, 1ULL << insert_index_list[cursor], insert_index_list[cursor]);
		}
		// flip controls
		basis_c_t0 ^= control_mask;
		// gather target
		ITYPE basis_c_t1 = basis_c_t0 ^ target_mask;
		// fetch values
		CTYPE cval_c_t0 = state[basis_c_t0];
		CTYPE cval_c_t1 = state[basis_c_t1];
		// set values
		state[basis_c_t0] = matrix[0] * cval_c_t0 + matrix[1] * cval_c_t1;
		state[basis_c_t1] = matrix[2] * cval_c_t0 + matrix[3] * cval_c_t1;
	}
	free(insert_index_list);
}
#endif

ITYPE* create_shift_mask_list(const UINT* array, UINT count, UINT target) {
	UINT size = count + 1;
	UINT* new_array = (UINT*)malloc(sizeof(UINT)*size);
	memcpy(new_array, array, sizeof(UINT)*count);
	new_array[count] = target;
	sort_ui(new_array, size);

	ITYPE* mask_array = (ITYPE*)malloc(sizeof(ITYPE)*size);
	for (UINT i = 0; i < size; ++i) {
		mask_array[i] = (1UL << new_array[i]) - 1;
	}
	free(new_array);
	return mask_array;
}

void multi_qubit_control_single_qubit_dense_matrix_gate_single(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count,
	UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	ITYPE* shift_masks = create_shift_mask_list(control_qubit_index_list, control_qubit_index_count, target_qubit_index);
	const ITYPE target_mask = 1ULL << target_qubit_index;
	ITYPE control_mask = create_control_mask(control_qubit_index_list, control_value_list, control_qubit_index_count);

	const UINT insert_index_list_count = control_qubit_index_count + 1;
	const ITYPE loop_dim = dim >> insert_index_list_count;

	ITYPE state_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create base index
		ITYPE basis_0 = state_index;
		for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
			basis_0 = (basis_0 & shift_masks[cursor]) + ((basis_0 & (~shift_masks[cursor])) << 1);
		}
		basis_0 += control_mask;
		ITYPE basis_1 = basis_0 ^ target_mask;

		// fetch values
		CTYPE cval0 = state[basis_0];
		CTYPE cval1 = state[basis_1];
		// set values
		state[basis_0] = matrix[0] * cval0 + matrix[1] * cval1;
		state[basis_1] = matrix[2] * cval0 + matrix[3] * cval1;
	}
	free(shift_masks);
}

void multi_qubit_control_single_qubit_dense_matrix_gate_single_stack(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count,
	UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {

	UINT sort_array[64];
	ITYPE mask_array[64];
	create_shift_mask_list_from_list_and_value_buf(control_qubit_index_list, control_qubit_index_count, target_qubit_index, sort_array, mask_array);
	const ITYPE target_mask = 1ULL << target_qubit_index;
	ITYPE control_mask = create_control_mask(control_qubit_index_list, control_value_list, control_qubit_index_count);

	const UINT insert_index_list_count = control_qubit_index_count + 1;
	const ITYPE loop_dim = dim >> insert_index_list_count;

	ITYPE state_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create base index
		ITYPE basis_0 = state_index;
		for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
			basis_0 = (basis_0 & mask_array[cursor]) + ((basis_0 & (~mask_array[cursor])) << 1);
		}
		basis_0 += control_mask;
		ITYPE basis_1 = basis_0 + target_mask;

		// fetch values
		CTYPE cval0 = state[basis_0];
		CTYPE cval1 = state[basis_1];
		// set values
		state[basis_0] = matrix[0] * cval0 + matrix[1] * cval1;
		state[basis_1] = matrix[2] * cval0 + matrix[3] * cval1;
	}
}
*/