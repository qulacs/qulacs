
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

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#ifdef _USE_SIMD
void double_qubit_dense_matrix_gate_simd_high(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim);
void double_qubit_dense_matrix_gate_simd_middle(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim);
void double_qubit_dense_matrix_gate_simd_low(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim);
#endif

void double_qubit_dense_matrix_gate_c(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim) {
#ifdef _OPENMP
	UINT threshold = 13;
	UINT default_thread_count = omp_get_max_threads();
	if (dim < (((ITYPE)1) << threshold)) omp_set_num_threads(1);
#endif

#ifdef _USE_SIMD
	double_qubit_dense_matrix_gate_simd(target_qubit_index1, target_qubit_index2, matrix, state, dim);
#else
	double_qubit_dense_matrix_gate_nosimd(target_qubit_index1, target_qubit_index2, matrix, state, dim);
#endif

#ifdef _OPENMP
	omp_set_num_threads(default_thread_count);
#endif
}

void double_qubit_dense_matrix_gate_nosimd(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim) {
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

#ifdef _OPENMP
#pragma omp parallel for
#endif
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



#ifdef _USE_SIMD
void double_qubit_dense_matrix_gate_simd_high(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim) {
	assert(target_qubit_index1 >= 2);
	assert(target_qubit_index2 >= 2);
	const UINT min_qubit_index = get_min_ui(target_qubit_index1, target_qubit_index2);
	const UINT max_qubit_index = get_max_ui(target_qubit_index1, target_qubit_index2);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	const ITYPE target_mask1_shift = 1ULL << (target_qubit_index1 + 1);
	const ITYPE target_mask2_shift = 1ULL << (target_qubit_index2 + 1);

	// loop variables
	const ITYPE loop_dim = dim / 4;
	ITYPE state_index;

	double* ptr_vec = (double*)vec;
	const double* ptr_mat = (const double*)mat;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; state_index += 4) {
		__m256d res_real_sum, res_imag_sum;
		__m256d vec_before, vec_after;
		__m256d vec_real00, vec_imag00;
		__m256d vec_real01, vec_imag01;
		__m256d vec_real10, vec_imag10;
		__m256d vec_real11, vec_imag11;
		__m256d dup_mr, dup_mi;

		// create index
		ITYPE basis00 = (state_index&low_mask)
			+ ((state_index&mid_mask) << 1)
			+ ((state_index&high_mask) << 2);
		// shited due to index from complex -> double
		basis00 = basis00 << 1;
		ITYPE basis01 = basis00 + target_mask1_shift;
		ITYPE basis10 = basis00 + target_mask2_shift;
		ITYPE basis11 = basis01 + target_mask2_shift;

		//// Pick 4 complex values from basis00
		vec_before = _mm256_loadu_pd(ptr_vec + basis00);	// (i1 r1 i0 r0)
		vec_after = _mm256_loadu_pd(ptr_vec + basis00 + 4);	// (i3 r3 i2 r2)
		//// Split real values and imag values via shuffle
		vec_real00 = _mm256_shuffle_pd(vec_before, vec_after, 0);	// (i1 r1 i0 r0) (i3 r3 i2 r2) -> (r3 r1 r2 r0) 0000 = 0
		vec_imag00 = _mm256_shuffle_pd(vec_before, vec_after, 15);		// (i1 r1 i0 r0) (i3 r3 i2 r2) -> (i3 i1 i2 i0) 1111 = 15
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[0]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[1]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_mul_pd(vec_real00, dup_mr);
		res_real_sum = _mm256_fnmadd_pd(vec_imag00, dup_mi, res_real_sum);	// -a*b+c
		res_imag_sum = _mm256_mul_pd(vec_real00, dup_mi);
		res_imag_sum = _mm256_fmadd_pd(vec_imag00, dup_mr, res_imag_sum);		// a*b+c

		//// Pick 4 complex values from basis01
		vec_before = _mm256_loadu_pd(ptr_vec + basis01);	// (i1 r1 i0 r0)
		vec_after = _mm256_loadu_pd(ptr_vec + basis01 + 4);	// (i3 r3 i2 r2)
		//// Split real values and imag values via shuffle
		vec_real01 = _mm256_shuffle_pd(vec_before, vec_after, 0);	// (i1 r1 i0 r0) (i3 r3 i2 r2) -> (r3 r1 r2 r0) 0000 = 0
		vec_imag01 = _mm256_shuffle_pd(vec_before, vec_after, 15);		// (i1 r1 i0 r0) (i3 r3 i2 r2) -> (i3 i1 i2 i0) 1111 = 15
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[2]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[3]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_fmadd_pd(vec_real01, dup_mr, res_real_sum);		// a*b+c
		res_real_sum = _mm256_fnmadd_pd(vec_imag01, dup_mi, res_real_sum);	//-a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_real01, dup_mi, res_imag_sum);		// a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_imag01, dup_mr, res_imag_sum);		// a*b+c

		//// Pick 4 complex values from basis10
		vec_before = _mm256_loadu_pd(ptr_vec + basis10);	// (i1 r1 i0 r0)
		vec_after = _mm256_loadu_pd(ptr_vec + basis10 + 4);	// (i3 r3 i2 r2)
		//// Split real values and imag values via shuffle
		vec_real10 = _mm256_shuffle_pd(vec_before, vec_after, 0);	// (i1 r1 i0 r0) (i3 r3 i2 r2) -> (r3 r1 r2 r0) 0000 = 0
		vec_imag10 = _mm256_shuffle_pd(vec_before, vec_after, 15);		// (i1 r1 i0 r0) (i3 r3 i2 r2) -> (i3 i1 i2 i0) 1111 = 15
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[4]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[5]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_fmadd_pd(vec_real10, dup_mr, res_real_sum);		// a*b+c
		res_real_sum = _mm256_fnmadd_pd(vec_imag10, dup_mi, res_real_sum);	//-a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_real10, dup_mi, res_imag_sum);		// a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_imag10, dup_mr, res_imag_sum);		// a*b+c

		//// Pick 4 complex values from basis11
		vec_before = _mm256_loadu_pd(ptr_vec + basis11);	// (i1 r1 i0 r0)
		vec_after = _mm256_loadu_pd(ptr_vec + basis11 + 4);	// (i3 r3 i2 r2)
		//// Split real values and imag values via shuffle
		vec_real11 = _mm256_shuffle_pd(vec_before, vec_after, 0);	// (i1 r1 i0 r0) (i3 r3 i2 r2) -> (r3 r1 r2 r0) 0000 = 0
		vec_imag11 = _mm256_shuffle_pd(vec_before, vec_after, 15);		// (i1 r1 i0 r0) (i3 r3 i2 r2) -> (i3 i1 i2 i0) 1111 = 15
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[6]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[7]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_fmadd_pd(vec_real11, dup_mr, res_real_sum);		// a*b+c
		res_real_sum = _mm256_fnmadd_pd(vec_imag11, dup_mi, res_real_sum);	//-a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_real11, dup_mi, res_imag_sum);		// a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_imag11, dup_mr, res_imag_sum);		// a*b+c

		//// Store
		vec_before = _mm256_shuffle_pd(res_real_sum, res_imag_sum, 0);	// (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r1 i1 i0 r0) 0000 = 0
		vec_after = _mm256_shuffle_pd(res_real_sum, res_imag_sum, 15);	// (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r3 i3 i2 r2) 1111 = 15
		_mm256_storeu_pd(ptr_vec + basis00, vec_before);
		_mm256_storeu_pd(ptr_vec + basis00 + 4, vec_after);


		// vector is already fetched, fetch successive matrix elements and perform dot(vec,vec) for other basis
		//// basis01
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[8]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[9]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_mul_pd(vec_real00, dup_mr);
		res_real_sum = _mm256_fnmadd_pd(vec_imag00, dup_mi, res_real_sum);	// -a*b+c
		res_imag_sum = _mm256_mul_pd(vec_real00, dup_mi);
		res_imag_sum = _mm256_fmadd_pd(vec_imag00, dup_mr, res_imag_sum);		// a*b+c
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[10]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[11]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_fmadd_pd(vec_real01, dup_mr, res_real_sum);		// a*b+c
		res_real_sum = _mm256_fnmadd_pd(vec_imag01, dup_mi, res_real_sum);	//-a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_real01, dup_mi, res_imag_sum);		// a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_imag01, dup_mr, res_imag_sum);		// a*b+c
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[12]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[13]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_fmadd_pd(vec_real10, dup_mr, res_real_sum);		// a*b+c
		res_real_sum = _mm256_fnmadd_pd(vec_imag10, dup_mi, res_real_sum);	//-a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_real10, dup_mi, res_imag_sum);		// a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_imag10, dup_mr, res_imag_sum);		// a*b+c
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[14]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[15]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_fmadd_pd(vec_real11, dup_mr, res_real_sum);		// a*b+c
		res_real_sum = _mm256_fnmadd_pd(vec_imag11, dup_mi, res_real_sum);	//-a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_real11, dup_mi, res_imag_sum);		// a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_imag11, dup_mr, res_imag_sum);		// a*b+c
		//// Store
		vec_before = _mm256_shuffle_pd(res_real_sum, res_imag_sum, 0);	// (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r1 i1 i0 r0) 0000 = 0
		vec_after = _mm256_shuffle_pd(res_real_sum, res_imag_sum, 15);	// (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r3 i3 i2 r2) 1111 = 15
		_mm256_storeu_pd(ptr_vec + basis01, vec_before);
		_mm256_storeu_pd(ptr_vec + basis01 + 4, vec_after);


		//// basis10
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[16]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[17]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_mul_pd(vec_real00, dup_mr);
		res_real_sum = _mm256_fnmadd_pd(vec_imag00, dup_mi, res_real_sum);	// -a*b+c
		res_imag_sum = _mm256_mul_pd(vec_real00, dup_mi);
		res_imag_sum = _mm256_fmadd_pd(vec_imag00, dup_mr, res_imag_sum);		// a*b+c
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[18]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[19]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_fmadd_pd(vec_real01, dup_mr, res_real_sum);		// a*b+c
		res_real_sum = _mm256_fnmadd_pd(vec_imag01, dup_mi, res_real_sum);	//-a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_real01, dup_mi, res_imag_sum);		// a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_imag01, dup_mr, res_imag_sum);		// a*b+c
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[20]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[21]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_fmadd_pd(vec_real10, dup_mr, res_real_sum);		// a*b+c
		res_real_sum = _mm256_fnmadd_pd(vec_imag10, dup_mi, res_real_sum);	//-a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_real10, dup_mi, res_imag_sum);		// a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_imag10, dup_mr, res_imag_sum);		// a*b+c
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[22]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[23]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_fmadd_pd(vec_real11, dup_mr, res_real_sum);		// a*b+c
		res_real_sum = _mm256_fnmadd_pd(vec_imag11, dup_mi, res_real_sum);	//-a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_real11, dup_mi, res_imag_sum);		// a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_imag11, dup_mr, res_imag_sum);		// a*b+c
		//// Store
		vec_before = _mm256_shuffle_pd(res_real_sum, res_imag_sum, 0);	// (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r1 i1 i0 r0) 0000 = 0
		vec_after = _mm256_shuffle_pd(res_real_sum, res_imag_sum, 15);	// (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r3 i3 i2 r2) 1111 = 15
		_mm256_storeu_pd(ptr_vec + basis10, vec_before);
		_mm256_storeu_pd(ptr_vec + basis10 + 4, vec_after);


		//// basis11
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[24]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[25]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_mul_pd(vec_real00, dup_mr);
		res_real_sum = _mm256_fnmadd_pd(vec_imag00, dup_mi, res_real_sum);	// -a*b+c
		res_imag_sum = _mm256_mul_pd(vec_real00, dup_mi);
		res_imag_sum = _mm256_fmadd_pd(vec_imag00, dup_mr, res_imag_sum);		// a*b+c
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[26]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[27]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_fmadd_pd(vec_real01, dup_mr, res_real_sum);		// a*b+c
		res_real_sum = _mm256_fnmadd_pd(vec_imag01, dup_mi, res_real_sum);	//-a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_real01, dup_mi, res_imag_sum);		// a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_imag01, dup_mr, res_imag_sum);		// a*b+c
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[28]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[29]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_fmadd_pd(vec_real10, dup_mr, res_real_sum);		// a*b+c
		res_real_sum = _mm256_fnmadd_pd(vec_imag10, dup_mi, res_real_sum);	//-a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_real10, dup_mi, res_imag_sum);		// a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_imag10, dup_mr, res_imag_sum);		// a*b+c
		//// Pick matrix elem with 4 dup
		dup_mr = _mm256_set1_pd(ptr_mat[30]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[31]);	// (mi0 mi0 mi0 mi0)
		//// Compute real and imag part
		res_real_sum = _mm256_fmadd_pd(vec_real11, dup_mr, res_real_sum);		// a*b+c
		res_real_sum = _mm256_fnmadd_pd(vec_imag11, dup_mi, res_real_sum);	//-a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_real11, dup_mi, res_imag_sum);		// a*b+c
		res_imag_sum = _mm256_fmadd_pd(vec_imag11, dup_mr, res_imag_sum);		// a*b+c
		//// Store
		vec_before = _mm256_shuffle_pd(res_real_sum, res_imag_sum, 0);	// (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r1 i1 i0 r0) 0000 = 0
		vec_after = _mm256_shuffle_pd(res_real_sum, res_imag_sum, 15);	// (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r3 i3 i2 r2) 1111 = 15
		_mm256_storeu_pd(ptr_vec + basis11, vec_before);
		_mm256_storeu_pd(ptr_vec + basis11 + 4, vec_after);
	}
}

void double_qubit_dense_matrix_gate_simd_low(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim) {
	assert(target_qubit_index1 < 2);
	assert(target_qubit_index2 < 2);
	assert(dim >= 8);

	// loop variables
	const ITYPE loop_dim = dim * 2;
	ITYPE state_index;

	double* ptr_vec = (double*)vec;
	const double* ptr_mat = (const double*)mat;
	if (target_qubit_index1 < target_qubit_index2) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (state_index = 0; state_index < loop_dim; state_index += 16) {
			__m256d vec1, vec2, vec3, vec4;
			__m256d u1, u2, u3, u4, u1f, u2f, u3f, u4f;
			__m256d mr, mi;

			vec1 = _mm256_loadu_pd(ptr_vec + state_index);		// c1 c0
			vec1 = _mm256_permute4x64_pd(vec1, 78); // (c1 c0) -> (c0 c1) : 1032 = 1*2 + 4*3 + 16*0 + 64*1 = 64+12+2=78

			vec2 = _mm256_loadu_pd(ptr_vec + state_index + 4);	// c3 c2
			vec2 = _mm256_permute4x64_pd(vec2, 78); // (c3 c2) -> (c2 c3) : 1032 = 1*2+4*3+16*0+32*1 = 46

			vec3 = _mm256_loadu_pd(ptr_vec + state_index + 8);	// c5 c4
			u1 = _mm256_blend_pd(vec1, vec3, 3); // (c0 c1) (c5 c4) -> (c0 c4) : 0011 = 3
			u2 = _mm256_blend_pd(vec1, vec3, 12);  // (c0 c1) (c5 c4) -> (c5 c1) : 1100 = 12
			u2 = _mm256_permute4x64_pd(u2, 78); // (c5 c1) -> (c1 c5) : 1032 = 1*2+4*3+16*0+64*1 = 64+12+2=78

			vec4 = _mm256_loadu_pd(ptr_vec + state_index + 12);	// c7 c6
			u3 = _mm256_blend_pd(vec2, vec4, 3); // (c2 c3) (c7 c6) -> (c2 c6) : 0011 = 3
			u4 = _mm256_blend_pd(vec2, vec4, 12);  // (c2 c3) (c7 c6) -> (c7 c3) : 1100 = 12
			u4 = _mm256_permute4x64_pd(u4, 78); // (c7 c3) -> (c3 c7) : 1032 = 1*2+4*3+16*0+32*1 = 46

			u1f = _mm256_permute4x64_pd(u1, 177); // 2301 = 64*2+16*3+1 = 128+48+1 = 177
			u2f = _mm256_permute4x64_pd(u2, 177);
			u3f = _mm256_permute4x64_pd(u3, 177);
			u4f = _mm256_permute4x64_pd(u4, 177);

			// u1  = (c0i c0r c4i c4r)
			// u2  = (c1i c1r c5i c5r)
			// u3  = (c2i c2r c6i c6r)
			// u4  = (c3i c3r c7i c7r)
			// u1f = (c0r c0i c4r c4i)
			// u2f = (c1r c1i c5r c5i)
			// u3f = (c2r c2i c6r c6i)
			// u4f = (c3r c3i c7r c7i)

			__m256d res_u1, res_u2, res_u3, res_u4, tmp_inv;
			tmp_inv = _mm256_set_pd(1, -1, 1, -1);

			mr = _mm256_set1_pd(ptr_mat[0]);
			res_u1 = _mm256_mul_pd(mr, u1); // c0i*m0r, -c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[1]);
			res_u1 = _mm256_fmaddsub_pd(mi, u1f, res_u1); // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[2]);
			res_u1 = _mm256_fmaddsub_pd(mr, u2, res_u1); // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r - m0i*c0i + c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[3]);
			res_u1 = _mm256_fmaddsub_pd(mi, u2f, res_u1); // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r, m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[4]);
			res_u1 = _mm256_fmaddsub_pd(mr, u3, res_u1);
			mi = _mm256_set1_pd(ptr_mat[5]);
			res_u1 = _mm256_fmaddsub_pd(mi, u3f, res_u1);
			mr = _mm256_set1_pd(ptr_mat[6]);
			res_u1 = _mm256_fmaddsub_pd(mr, u4, res_u1);
			mi = _mm256_set1_pd(ptr_mat[7]);
			res_u1 = _mm256_fmaddsub_pd(mi, u4f, res_u1);
			res_u1 = _mm256_mul_pd(res_u1, tmp_inv);

			mr = _mm256_set1_pd(ptr_mat[8]);
			res_u2 = _mm256_mul_pd(mr, u1); // c0i*m0r, -c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[9]);
			res_u2 = _mm256_fmaddsub_pd(mi, u1f, res_u2); // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[10]);
			res_u2 = _mm256_fmaddsub_pd(mr, u2, res_u2); // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r - m0i*c0i + c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[11]);
			res_u2 = _mm256_fmaddsub_pd(mi, u2f, res_u2); // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r, m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[12]);
			res_u2 = _mm256_fmaddsub_pd(mr, u3, res_u2);
			mi = _mm256_set1_pd(ptr_mat[13]);
			res_u2 = _mm256_fmaddsub_pd(mi, u3f, res_u2);
			mr = _mm256_set1_pd(ptr_mat[14]);
			res_u2 = _mm256_fmaddsub_pd(mr, u4, res_u2);
			mi = _mm256_set1_pd(ptr_mat[15]);
			res_u2 = _mm256_fmaddsub_pd(mi, u4f, res_u2);
			res_u2 = _mm256_mul_pd(res_u2, tmp_inv);

			res_u2 = _mm256_permute4x64_pd(res_u2, 78); // flip
			vec1 = _mm256_blend_pd(res_u1, res_u2, 3); // blend
			vec2 = _mm256_blend_pd(res_u1, res_u2, 12); // blend
			vec1 = _mm256_permute4x64_pd(vec1, 78); // flip
			_mm256_storeu_pd(ptr_vec + state_index, vec1);
			_mm256_storeu_pd(ptr_vec + state_index + 8, vec2);


			mr = _mm256_set1_pd(ptr_mat[16]);
			res_u3 = _mm256_mul_pd(mr, u1); // c0i*m0r, -c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[17]);
			res_u3 = _mm256_fmaddsub_pd(mi, u1f, res_u3); // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[18]);
			res_u3 = _mm256_fmaddsub_pd(mr, u2, res_u3); // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r - m0i*c0i + c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[19]);
			res_u3 = _mm256_fmaddsub_pd(mi, u2f, res_u3); // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r, m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[20]);
			res_u3 = _mm256_fmaddsub_pd(mr, u3, res_u3);
			mi = _mm256_set1_pd(ptr_mat[21]);
			res_u3 = _mm256_fmaddsub_pd(mi, u3f, res_u3);
			mr = _mm256_set1_pd(ptr_mat[22]);
			res_u3 = _mm256_fmaddsub_pd(mr, u4, res_u3);
			mi = _mm256_set1_pd(ptr_mat[23]);
			res_u3 = _mm256_fmaddsub_pd(mi, u4f, res_u3);
			res_u3 = _mm256_mul_pd(res_u3, tmp_inv);

			mr = _mm256_set1_pd(ptr_mat[24]);
			res_u4 = _mm256_mul_pd(mr, u1); // c0i*m0r, -c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[25]);
			res_u4 = _mm256_fmaddsub_pd(mi, u1f, res_u4); // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[26]);
			res_u4 = _mm256_fmaddsub_pd(mr, u2, res_u4); // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r - m0i*c0i + c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[27]);
			res_u4 = _mm256_fmaddsub_pd(mi, u2f, res_u4); // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r, m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[28]);
			res_u4 = _mm256_fmaddsub_pd(mr, u3, res_u4);
			mi = _mm256_set1_pd(ptr_mat[29]);
			res_u4 = _mm256_fmaddsub_pd(mi, u3f, res_u4);
			mr = _mm256_set1_pd(ptr_mat[30]);
			res_u4 = _mm256_fmaddsub_pd(mr, u4, res_u4);
			mi = _mm256_set1_pd(ptr_mat[31]);
			res_u4 = _mm256_fmaddsub_pd(mi, u4f, res_u4);
			res_u4 = _mm256_mul_pd(res_u4, tmp_inv);

			res_u4 = _mm256_permute4x64_pd(res_u4, 78); // flip
			vec3 = _mm256_blend_pd(res_u3, res_u4, 3); // blend
			vec4 = _mm256_blend_pd(res_u3, res_u4, 12); // blend
			vec3 = _mm256_permute4x64_pd(vec3, 78); // flip
			_mm256_storeu_pd(ptr_vec + state_index + 4, vec3);
			_mm256_storeu_pd(ptr_vec + state_index + 12, vec4);
		}
	}
	else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; state_index += 16) {
			__m256d vec1, vec2, vec3, vec4;
			__m256d u1, u2, u3, u4, u1f, u2f, u3f, u4f;
			__m256d mr, mi;

			vec1 = _mm256_loadu_pd(ptr_vec + state_index);		// c1 c0
			vec1 = _mm256_permute4x64_pd(vec1, 78); // (c1 c0) -> (c0 c1) : 1032 = 1*2 + 4*3 + 16*0 + 64*1 = 64+12+2=78

			vec2 = _mm256_loadu_pd(ptr_vec + state_index + 4);	// c3 c2
			vec2 = _mm256_permute4x64_pd(vec2, 78); // (c3 c2) -> (c2 c3) : 1032 = 1*2+4*3+16*0+32*1 = 46

			vec3 = _mm256_loadu_pd(ptr_vec + state_index + 8);	// c5 c4
			u1 = _mm256_blend_pd(vec1, vec3, 3); // (c0 c1) (c5 c4) -> (c0 c4) : 0011 = 3
			u2 = _mm256_blend_pd(vec1, vec3, 12);  // (c0 c1) (c5 c4) -> (c5 c1) : 1100 = 12
			u2 = _mm256_permute4x64_pd(u2, 78); // (c5 c1) -> (c1 c5) : 1032 = 1*2+4*3+16*0+64*1 = 64+12+2=78

			vec4 = _mm256_loadu_pd(ptr_vec + state_index + 12);	// c7 c6
			u3 = _mm256_blend_pd(vec2, vec4, 3); // (c2 c3) (c7 c6) -> (c2 c6) : 0011 = 3
			u4 = _mm256_blend_pd(vec2, vec4, 12);  // (c2 c3) (c7 c6) -> (c7 c3) : 1100 = 12
			u4 = _mm256_permute4x64_pd(u4, 78); // (c7 c3) -> (c3 c7) : 1032 = 1*2+4*3+16*0+32*1 = 46

			u1f = _mm256_permute4x64_pd(u1, 177); // 2301 = 64*2+16*3+1 = 128+48+1 = 177
			u2f = _mm256_permute4x64_pd(u2, 177);
			u3f = _mm256_permute4x64_pd(u3, 177);
			u4f = _mm256_permute4x64_pd(u4, 177);

			// u1  = (c0i c0r c4i c4r)
			// u2  = (c1i c1r c5i c5r)
			// u3  = (c2i c2r c6i c6r)
			// u4  = (c3i c3r c7i c7r)
			// u1f = (c0r c0i c4r c4i)
			// u2f = (c1r c1i c5r c5i)
			// u3f = (c2r c2i c6r c6i)
			// u4f = (c3r c3i c7r c7i)

			__m256d res_u1, res_u2, res_u3, res_u4, tmp_inv;
			tmp_inv = _mm256_set_pd(1, -1, 1, -1);

			mr = _mm256_set1_pd(ptr_mat[0]);
			res_u1 = _mm256_mul_pd(mr, u1); // c0i*m0r, -c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[1]);
			res_u1 = _mm256_fmaddsub_pd(mi, u1f, res_u1); // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[2]);
			res_u1 = _mm256_fmaddsub_pd(mr, u3, res_u1); // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r - m0i*c0i + c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[3]);
			res_u1 = _mm256_fmaddsub_pd(mi, u3f, res_u1); // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r, m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[4]);
			res_u1 = _mm256_fmaddsub_pd(mr, u2, res_u1);
			mi = _mm256_set1_pd(ptr_mat[5]);
			res_u1 = _mm256_fmaddsub_pd(mi, u2f, res_u1);
			mr = _mm256_set1_pd(ptr_mat[6]);
			res_u1 = _mm256_fmaddsub_pd(mr, u4, res_u1);
			mi = _mm256_set1_pd(ptr_mat[7]);
			res_u1 = _mm256_fmaddsub_pd(mi, u4f, res_u1);
			res_u1 = _mm256_mul_pd(res_u1, tmp_inv);

			mr = _mm256_set1_pd(ptr_mat[16]);
			res_u3 = _mm256_mul_pd(mr, u1); // c0i*m0r, -c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[17]);
			res_u3 = _mm256_fmaddsub_pd(mi, u1f, res_u3); // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[18]);
			res_u3 = _mm256_fmaddsub_pd(mr, u3, res_u3); // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r - m0i*c0i + c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[19]);
			res_u3 = _mm256_fmaddsub_pd(mi, u3f, res_u3); // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r, m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[20]);
			res_u3 = _mm256_fmaddsub_pd(mr, u2, res_u3);
			mi = _mm256_set1_pd(ptr_mat[21]);
			res_u3 = _mm256_fmaddsub_pd(mi, u2f, res_u3);
			mr = _mm256_set1_pd(ptr_mat[22]);
			res_u3 = _mm256_fmaddsub_pd(mr, u4, res_u3);
			mi = _mm256_set1_pd(ptr_mat[23]);
			res_u3 = _mm256_fmaddsub_pd(mi, u4f, res_u3);
			res_u3 = _mm256_mul_pd(res_u3, tmp_inv);

			res_u3 = _mm256_permute4x64_pd(res_u3, 78); // flip
			vec1 = _mm256_blend_pd(res_u1, res_u3, 3); // blend
			vec3 = _mm256_blend_pd(res_u1, res_u3, 12); // blend
			vec1 = _mm256_permute4x64_pd(vec1, 78); // flip
			_mm256_storeu_pd(ptr_vec + state_index, vec1);
			_mm256_storeu_pd(ptr_vec + state_index + 8, vec3);

			mr = _mm256_set1_pd(ptr_mat[8]);
			res_u2 = _mm256_mul_pd(mr, u1); // c0i*m0r, -c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[9]);
			res_u2 = _mm256_fmaddsub_pd(mi, u1f, res_u2); // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[10]);
			res_u2 = _mm256_fmaddsub_pd(mr, u3, res_u2); // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r - m0i*c0i + c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[11]);
			res_u2 = _mm256_fmaddsub_pd(mi, u3f, res_u2); // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r, m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[12]);
			res_u2 = _mm256_fmaddsub_pd(mr, u2, res_u2);
			mi = _mm256_set1_pd(ptr_mat[13]);
			res_u2 = _mm256_fmaddsub_pd(mi, u2f, res_u2);
			mr = _mm256_set1_pd(ptr_mat[14]);
			res_u2 = _mm256_fmaddsub_pd(mr, u4, res_u2);
			mi = _mm256_set1_pd(ptr_mat[15]);
			res_u2 = _mm256_fmaddsub_pd(mi, u4f, res_u2);
			res_u2 = _mm256_mul_pd(res_u2, tmp_inv);

			mr = _mm256_set1_pd(ptr_mat[24]);
			res_u4 = _mm256_mul_pd(mr, u1); // c0i*m0r, -c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[25]);
			res_u4 = _mm256_fmaddsub_pd(mi, u1f, res_u4); // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[26]);
			res_u4 = _mm256_fmaddsub_pd(mr, u3, res_u4); // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r - m0i*c0i + c0r*m0r
			mi = _mm256_set1_pd(ptr_mat[27]);
			res_u4 = _mm256_fmaddsub_pd(mi, u3f, res_u4); // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r, m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
			mr = _mm256_set1_pd(ptr_mat[28]);
			res_u4 = _mm256_fmaddsub_pd(mr, u2, res_u4);
			mi = _mm256_set1_pd(ptr_mat[29]);
			res_u4 = _mm256_fmaddsub_pd(mi, u2f, res_u4);
			mr = _mm256_set1_pd(ptr_mat[30]);
			res_u4 = _mm256_fmaddsub_pd(mr, u4, res_u4);
			mi = _mm256_set1_pd(ptr_mat[31]);
			res_u4 = _mm256_fmaddsub_pd(mi, u4f, res_u4);
			res_u4 = _mm256_mul_pd(res_u4, tmp_inv);

			res_u4 = _mm256_permute4x64_pd(res_u4, 78); // flip
			vec2 = _mm256_blend_pd(res_u2, res_u4, 3); // blend
			vec4 = _mm256_blend_pd(res_u2, res_u4, 12); // blend
			vec2 = _mm256_permute4x64_pd(vec2, 78); // flip
			_mm256_storeu_pd(ptr_vec + state_index + 4, vec2);
			_mm256_storeu_pd(ptr_vec + state_index + 12, vec4);
		}
	}
}


__inline void _element_swap(CTYPE* vec, UINT i1, UINT i2) {
	CTYPE temp = vec[i1];
	vec[i1] = vec[i2];
	vec[i2] = temp;
}

void double_qubit_dense_matrix_gate_simd_middle(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE _mat[16], CTYPE* vec, ITYPE dim) {
	CTYPE mat[16];
	memcpy(mat, _mat, sizeof(CTYPE) * 16);
	if (target_qubit_index2 < target_qubit_index1) {
		UINT temp = target_qubit_index1;
		target_qubit_index1 = target_qubit_index2;
		target_qubit_index2 = temp;
		_element_swap(mat, 1, 2);
		_element_swap(mat, 4, 8);
		_element_swap(mat, 7, 11);
		_element_swap(mat, 13, 14);
		_element_swap(mat, 5, 10);
		_element_swap(mat, 6, 9);
	}
	assert(target_qubit_index1 < 2);
	assert(target_qubit_index2 >= 2);

	const UINT min_qubit_index = get_min_ui(target_qubit_index1, target_qubit_index2);
	const UINT max_qubit_index = get_max_ui(target_qubit_index1, target_qubit_index2);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	const ITYPE target_mask1_shift = 1ULL << (target_qubit_index1 + 1);
	const ITYPE target_mask2_shift = 1ULL << (target_qubit_index2 + 1);

	// loop variables
	const ITYPE loop_dim = dim / 4;
	ITYPE state_index;

	double* ptr_vec = (double*)vec;
	const double* ptr_mat = (const double*)mat;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; state_index+=2) {
		// create index
		ITYPE basis00 = (state_index&low_mask)
			+ ((state_index&mid_mask) << 1)
			+ ((state_index&high_mask) << 2);
		// shited due to index from complex -> double
		basis00 = basis00 << 1;
		//ITYPE basis01 = basis00 + target_mask1_shift;
		ITYPE basis10 = basis00 + target_mask2_shift;
		//ITYPE basis11 = basis01 + target_mask2_shift;

		//// Pick 4 complex values from basis00
		__m256d vec_bef0, vec_aft0, vec_bef1, vec_aft1;
		vec_bef0 = _mm256_loadu_pd(ptr_vec + basis00);		// (i1 r1 i0 r0)
		vec_aft0 = _mm256_loadu_pd(ptr_vec + basis00 + 4);	// (i3 r3 i2 r2)
		vec_bef1 = _mm256_loadu_pd(ptr_vec + basis10);
		vec_aft1 = _mm256_loadu_pd(ptr_vec + basis10 + 4);

		__m256d vec_u0, vec_u1, vec_u2, vec_u3;
		__m256d vec_u0f, vec_u1f, vec_u2f, vec_u3f;
		__m256d vec_inv;
		vec_inv = _mm256_set_pd(1, -1, 1, -1);
		if (target_qubit_index1 == 0) {
			vec_aft0 = _mm256_permute4x64_pd(vec_aft0, 78); // (3 2 1 0) -> (1 0 3 2) 1*2 + 4*3 + 16*0 + 64*1 = 64+12+2 = 78
			vec_aft1 = _mm256_permute4x64_pd(vec_aft1, 78);
			vec_u0 = _mm256_blend_pd(vec_bef0, vec_aft0, 12); // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
			vec_u1 = _mm256_blend_pd(vec_bef0, vec_aft0, 3); // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
			vec_u2 = _mm256_blend_pd(vec_bef1, vec_aft1, 12); // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
			vec_u3 = _mm256_blend_pd(vec_bef1, vec_aft1, 3); // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
			vec_u1 = _mm256_permute4x64_pd(vec_u1, 78); // (3 2 1 0) -> (1 0 3 2) 1*2 + 4*3 + 16*0 + 64*1 = 64+12+2 = 78
			vec_u3 = _mm256_permute4x64_pd(vec_u3, 78);
		}
		else {
			vec_u0 = vec_bef0;
			vec_u1 = vec_aft0;
			vec_u2 = vec_bef1;
			vec_u3 = vec_aft1;
		}
		vec_u0f = _mm256_permute_pd(vec_u0, 5); // 1*1 + 2*0 + 4*1 + 8*0
		vec_u1f = _mm256_permute_pd(vec_u1, 5); // 1*1 + 2*0 + 4*1 + 8*0
		vec_u2f = _mm256_permute_pd(vec_u2, 5); // 1*1 + 2*0 + 4*1 + 8*0
		vec_u3f = _mm256_permute_pd(vec_u3, 5); // 1*1 + 2*0 + 4*1 + 8*0
		vec_u0f = _mm256_mul_pd(vec_u0f, vec_inv);
		vec_u1f = _mm256_mul_pd(vec_u1f, vec_inv);
		vec_u2f = _mm256_mul_pd(vec_u2f, vec_inv);
		vec_u3f = _mm256_mul_pd(vec_u3f, vec_inv);

		__m256d dup_mr, dup_mi;

		__m256d res_sum0, res_sum1, res_sum2, res_sum3;
		dup_mr = _mm256_set1_pd(ptr_mat[0]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[1]);	// (mi0 mi0 mi0 mi0)
		res_sum0 = _mm256_mul_pd(vec_u0, dup_mr);
		res_sum0 = _mm256_fmadd_pd(vec_u0f, dup_mi, res_sum0);
		dup_mr = _mm256_set1_pd(ptr_mat[2]);	// (mr1 mr1 mr1 mr1)
		dup_mi = _mm256_set1_pd(ptr_mat[3]);	// (mi1 mi1 mi1 mi1)
		res_sum0 = _mm256_fmadd_pd(vec_u1, dup_mr, res_sum0);
		res_sum0 = _mm256_fmadd_pd(vec_u1f, dup_mi, res_sum0);
		dup_mr = _mm256_set1_pd(ptr_mat[4]);	// (mr1 mr1 mr1 mr1)
		dup_mi = _mm256_set1_pd(ptr_mat[5]);	// (mi1 mi1 mi1 mi1)
		res_sum0 = _mm256_fmadd_pd(vec_u2, dup_mr, res_sum0);
		res_sum0 = _mm256_fmadd_pd(vec_u2f, dup_mi, res_sum0);
		dup_mr = _mm256_set1_pd(ptr_mat[6]);	// (mr1 mr1 mr1 mr1)
		dup_mi = _mm256_set1_pd(ptr_mat[7]);	// (mi1 mi1 mi1 mi1)
		res_sum0 = _mm256_fmadd_pd(vec_u3, dup_mr, res_sum0);
		res_sum0 = _mm256_fmadd_pd(vec_u3f, dup_mi, res_sum0);

		dup_mr = _mm256_set1_pd(ptr_mat[8]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[9]);	// (mi0 mi0 mi0 mi0)
		res_sum1 = _mm256_mul_pd(vec_u0, dup_mr);
		res_sum1 = _mm256_fmadd_pd(vec_u0f, dup_mi, res_sum1);
		dup_mr = _mm256_set1_pd(ptr_mat[10]);	// (mr1 mr1 mr1 mr1)
		dup_mi = _mm256_set1_pd(ptr_mat[11]);	// (mi1 mi1 mi1 mi1)
		res_sum1 = _mm256_fmadd_pd(vec_u1, dup_mr, res_sum1);
		res_sum1 = _mm256_fmadd_pd(vec_u1f, dup_mi, res_sum1);
		dup_mr = _mm256_set1_pd(ptr_mat[12]);	// (mr1 mr1 mr1 mr1)
		dup_mi = _mm256_set1_pd(ptr_mat[13]);	// (mi1 mi1 mi1 mi1)
		res_sum1 = _mm256_fmadd_pd(vec_u2, dup_mr, res_sum1);
		res_sum1 = _mm256_fmadd_pd(vec_u2f, dup_mi, res_sum1);
		dup_mr = _mm256_set1_pd(ptr_mat[14]);	// (mr1 mr1 mr1 mr1)
		dup_mi = _mm256_set1_pd(ptr_mat[15]);	// (mi1 mi1 mi1 mi1)
		res_sum1 = _mm256_fmadd_pd(vec_u3, dup_mr, res_sum1);
		res_sum1 = _mm256_fmadd_pd(vec_u3f, dup_mi, res_sum1);

		dup_mr = _mm256_set1_pd(ptr_mat[16]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[17]);	// (mi0 mi0 mi0 mi0)
		res_sum2 = _mm256_mul_pd(vec_u0, dup_mr);
		res_sum2 = _mm256_fmadd_pd(vec_u0f, dup_mi, res_sum2);
		dup_mr = _mm256_set1_pd(ptr_mat[18]);	// (mr1 mr1 mr1 mr1)
		dup_mi = _mm256_set1_pd(ptr_mat[19]);	// (mi1 mi1 mi1 mi1)
		res_sum2 = _mm256_fmadd_pd(vec_u1, dup_mr, res_sum2);
		res_sum2 = _mm256_fmadd_pd(vec_u1f, dup_mi, res_sum2);
		dup_mr = _mm256_set1_pd(ptr_mat[20]);	// (mr1 mr1 mr1 mr1)
		dup_mi = _mm256_set1_pd(ptr_mat[21]);	// (mi1 mi1 mi1 mi1)
		res_sum2 = _mm256_fmadd_pd(vec_u2, dup_mr, res_sum2);
		res_sum2 = _mm256_fmadd_pd(vec_u2f, dup_mi, res_sum2);
		dup_mr = _mm256_set1_pd(ptr_mat[22]);	// (mr1 mr1 mr1 mr1)
		dup_mi = _mm256_set1_pd(ptr_mat[23]);	// (mi1 mi1 mi1 mi1)
		res_sum2 = _mm256_fmadd_pd(vec_u3, dup_mr, res_sum2);
		res_sum2 = _mm256_fmadd_pd(vec_u3f, dup_mi, res_sum2);

		dup_mr = _mm256_set1_pd(ptr_mat[24]);	// (mr0 mr0 mr0 mr0)
		dup_mi = _mm256_set1_pd(ptr_mat[25]);	// (mi0 mi0 mi0 mi0)
		res_sum3 = _mm256_mul_pd(vec_u0, dup_mr);
		res_sum3 = _mm256_fmadd_pd(vec_u0f, dup_mi, res_sum3);
		dup_mr = _mm256_set1_pd(ptr_mat[26]);	// (mr1 mr1 mr1 mr1)
		dup_mi = _mm256_set1_pd(ptr_mat[27]);	// (mi1 mi1 mi1 mi1)
		res_sum3 = _mm256_fmadd_pd(vec_u1, dup_mr, res_sum3);
		res_sum3 = _mm256_fmadd_pd(vec_u1f, dup_mi, res_sum3);
		dup_mr = _mm256_set1_pd(ptr_mat[28]);	// (mr1 mr1 mr1 mr1)
		dup_mi = _mm256_set1_pd(ptr_mat[29]);	// (mi1 mi1 mi1 mi1)
		res_sum3 = _mm256_fmadd_pd(vec_u2, dup_mr, res_sum3);
		res_sum3 = _mm256_fmadd_pd(vec_u2f, dup_mi, res_sum3);
		dup_mr = _mm256_set1_pd(ptr_mat[30]);	// (mr1 mr1 mr1 mr1)
		dup_mi = _mm256_set1_pd(ptr_mat[31]);	// (mi1 mi1 mi1 mi1)
		res_sum3 = _mm256_fmadd_pd(vec_u3, dup_mr, res_sum3);
		res_sum3 = _mm256_fmadd_pd(vec_u3f, dup_mi, res_sum3);

		if (target_qubit_index1 == 0) {
			res_sum1 = _mm256_permute4x64_pd(res_sum1, 78); // (3 2 1 0) -> (1 0 3 2) 1*2 + 4*3 + 16*0 + 64*1 = 64+12+2 = 78
			res_sum3 = _mm256_permute4x64_pd(res_sum3, 78);
			vec_bef0 = _mm256_blend_pd(res_sum0, res_sum1, 12); // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
			vec_aft0 = _mm256_blend_pd(res_sum0, res_sum1, 3); // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
			vec_bef1 = _mm256_blend_pd(res_sum2, res_sum3, 12); // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
			vec_aft1 = _mm256_blend_pd(res_sum2, res_sum3, 3); // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
			vec_aft0 = _mm256_permute4x64_pd(vec_aft0, 78); // (3 2 1 0) -> (1 0 3 2) 1*2 + 4*3 + 16*0 + 64*1 = 64+12+2 = 78
			vec_aft1 = _mm256_permute4x64_pd(vec_aft1, 78);
		}
		else {
			vec_bef0 = res_sum0;
			vec_aft0 = res_sum1;
			vec_bef1 = res_sum2;
			vec_aft1 = res_sum3;
		}
		//// Store
		_mm256_storeu_pd(ptr_vec + basis00, vec_bef0);		// (i1 r1 i0 r0)
		_mm256_storeu_pd(ptr_vec + basis00 + 4, vec_aft0);	// (i3 r3 i2 r2)
		_mm256_storeu_pd(ptr_vec + basis10, vec_bef1);
		_mm256_storeu_pd(ptr_vec + basis10 + 4, vec_aft1);
	}
}

void double_qubit_dense_matrix_gate_simd(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim) {
	assert(target_qubit_index1 != target_qubit_index2);
	if (dim == 4) {
		// avx2 code cannot use for 2-qubit state
		double_qubit_dense_matrix_gate_nosimd(target_qubit_index1, target_qubit_index2, mat, vec, dim);
	}
	else if (target_qubit_index1 >= 2 && target_qubit_index2 >= 2) {
		double_qubit_dense_matrix_gate_simd_high(target_qubit_index1, target_qubit_index2, mat, vec, dim);
	}
	else if (target_qubit_index1 >= 2 || target_qubit_index2 >= 2) {
		double_qubit_dense_matrix_gate_simd_middle(target_qubit_index1, target_qubit_index2, mat, vec, dim);
	}
	else {
		double_qubit_dense_matrix_gate_simd_low(target_qubit_index1, target_qubit_index2, mat, vec, dim);
	}
}
#endif
