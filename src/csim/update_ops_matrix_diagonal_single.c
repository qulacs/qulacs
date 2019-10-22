
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

//void single_qubit_diagonal_matrix_gate_old_single(UINT target_qubit_index, const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim);
//void single_qubit_diagonal_matrix_gate_old_parallel(UINT target_qubit_index, const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim);

void single_qubit_diagonal_matrix_gate(UINT target_qubit_index, const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
	//single_qubit_diagonal_matrix_gate_old_single(target_qubit_index, diagonal_matrix, state, dim);
	//single_qubit_diagonal_matrix_gate_old_parallel(target_qubit_index, diagonal_matrix, state, dim);
	//single_qubit_diagonal_matrix_gate_single_unroll(target_qubit_index, diagonal_matrix, state, dim);
	//single_qubit_diagonal_matrix_gate_single_simd(target_qubit_index, diagonal_matrix, state, dim);
	//single_qubit_diagonal_matrix_gate_parallel_simd(target_qubit_index, diagonal_matrix, state, dim);

#ifdef _USE_SIMD
#ifdef _OPENMP
	UINT threshold = 12;
	if (dim < (((ITYPE)1) << threshold)) {
		single_qubit_diagonal_matrix_gate_single_simd(target_qubit_index, diagonal_matrix, state, dim);
	}
	else {
		single_qubit_diagonal_matrix_gate_parallel_simd(target_qubit_index, diagonal_matrix, state, dim);
	}
#else
	single_qubit_diagonal_matrix_gate_single_simd(target_qubit_index, diagonal_matrix, state, dim);
#endif
#else
#ifdef _OPENMP
	UINT threshold = 12;
	if (dim < (((ITYPE)1) << threshold)) {
		single_qubit_diagonal_matrix_gate_single_unroll(target_qubit_index, diagonal_matrix, state, dim);
	}
	else {
		single_qubit_diagonal_matrix_gate_parallel_unroll(target_qubit_index, diagonal_matrix, state, dim);
	}
#else
	single_qubit_diagonal_matrix_gate_single_unroll(target_qubit_index, diagonal_matrix, state, dim);
#endif
#endif
}

void single_qubit_diagonal_matrix_gate_single_unroll(UINT target_qubit_index, const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
	// loop variables
	const ITYPE loop_dim = dim;
	ITYPE state_index;
	if (target_qubit_index == 0) {
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			state[state_index] *= diagonal_matrix[0];
			state[state_index+1] *= diagonal_matrix[1];
		}
	}
	else {
		ITYPE mask = 1ULL << target_qubit_index;
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			int bitval = ((state_index&mask) != 0);
			state[state_index] *= diagonal_matrix[bitval];
			state[state_index + 1] *= diagonal_matrix[bitval];
		}
	}
}

#ifdef _OPENMP
void single_qubit_diagonal_matrix_gate_parallel_unroll(UINT target_qubit_index, const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
	// loop variables
	const ITYPE loop_dim = dim;
	ITYPE state_index;
	if (target_qubit_index == 0) {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			state[state_index] *= diagonal_matrix[0];
			state[state_index + 1] *= diagonal_matrix[1];
		}
	}
	else {
		ITYPE mask = 1ULL << target_qubit_index;
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			int bitval = ((state_index&mask) != 0);
			state[state_index] *= diagonal_matrix[bitval];
			state[state_index + 1] *= diagonal_matrix[bitval];
		}
	}
}
#endif

#ifdef _USE_SIMD
void single_qubit_diagonal_matrix_gate_single_simd(UINT target_qubit_index, const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
	// loop variables
	const ITYPE loop_dim = dim;
	ITYPE state_index;
	if (target_qubit_index == 0) {
		__m256d mv0 = _mm256_set_pd(-cimag(diagonal_matrix[1]),creal(diagonal_matrix[1]),-cimag(diagonal_matrix[0]),creal(diagonal_matrix[0]));
		__m256d mv1 = _mm256_set_pd(creal(diagonal_matrix[1]), cimag(diagonal_matrix[1]), creal(diagonal_matrix[0]), cimag(diagonal_matrix[0]));
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			double* ptr = (double*)(state + state_index);
			__m256d data = _mm256_loadu_pd(ptr);
			__m256d data0 = _mm256_mul_pd(data, mv0);
			__m256d data1 = _mm256_mul_pd(data, mv1);
			data = _mm256_hadd_pd(data0, data1);
			_mm256_storeu_pd(ptr, data);
		}
	}
	else {
		__m256d mv0 = _mm256_set_pd(-cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]), -cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]));
		__m256d mv1 = _mm256_set_pd(creal(diagonal_matrix[0]), cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]), cimag(diagonal_matrix[0]));
		__m256d mv2 = _mm256_set_pd(-cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]), -cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]));
		__m256d mv3 = _mm256_set_pd(creal(diagonal_matrix[1]), cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]), cimag(diagonal_matrix[1]));
		//__m256i mask = _mm256_set1_epi64x(1LL<<target_qubit_index);
		ITYPE mask = 1LL << target_qubit_index;
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			double* ptr = (double*)(state + state_index);
			ITYPE flag = (state_index & mask);
			__m256d mv4 = flag ? mv2 : mv0;
			__m256d mv5 = flag ? mv3 : mv1;
			__m256d data = _mm256_loadu_pd(ptr);
			__m256d data0 = _mm256_mul_pd(data, mv4);
			__m256d data1 = _mm256_mul_pd(data, mv5);
			data = _mm256_hadd_pd(data0, data1);
			_mm256_storeu_pd(ptr, data);
		}
	}
}

#ifdef _OPENMP
void single_qubit_diagonal_matrix_gate_parallel_simd(UINT target_qubit_index, const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
	// loop variables
	const ITYPE loop_dim = dim;
	ITYPE state_index;
	if (target_qubit_index == 0) {
		__m256d mv0 = _mm256_set_pd(-cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]), -cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]));
		__m256d mv1 = _mm256_set_pd(creal(diagonal_matrix[1]), cimag(diagonal_matrix[1]), creal(diagonal_matrix[0]), cimag(diagonal_matrix[0]));
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			double* ptr = (double*)(state + state_index);
			__m256d data = _mm256_loadu_pd(ptr);
			__m256d data0 = _mm256_mul_pd(data, mv0);
			__m256d data1 = _mm256_mul_pd(data, mv1);
			data = _mm256_hadd_pd(data0, data1);
			_mm256_storeu_pd(ptr, data);
		}
	}
	else {
		__m256d mv0 = _mm256_set_pd(-cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]), -cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]));
		__m256d mv1 = _mm256_set_pd(creal(diagonal_matrix[0]), cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]), cimag(diagonal_matrix[0]));
		__m256d mv2 = _mm256_set_pd(-cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]), -cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]));
		__m256d mv3 = _mm256_set_pd(creal(diagonal_matrix[1]), cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]), cimag(diagonal_matrix[1]));
		//__m256i mask = _mm256_set1_epi64x(1LL<<target_qubit_index);
		ITYPE mask = 1LL << target_qubit_index;
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			double* ptr = (double*)(state + state_index);
			ITYPE flag = (state_index & mask);
			__m256d mv4 = flag ? mv2 : mv0;
			__m256d mv5 = flag ? mv3 : mv1;
			__m256d data = _mm256_loadu_pd(ptr);
			__m256d data0 = _mm256_mul_pd(data, mv4);
			__m256d data1 = _mm256_mul_pd(data, mv5);
			data = _mm256_hadd_pd(data0, data1);
			_mm256_storeu_pd(ptr, data);
		}
	}
}
#endif
#endif


/*

void single_qubit_diagonal_matrix_gate_old_single(UINT target_qubit_index, const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
	// loop variables
	const ITYPE loop_dim = dim;
	ITYPE state_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// determin matrix pos
		UINT bit_val = (state_index >> target_qubit_index) % 2;

		// set value
		state[state_index] *= diagonal_matrix[bit_val];
	}
}

#ifdef _OPENMP
void single_qubit_diagonal_matrix_gate_old_parallel(UINT target_qubit_index, const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
	// loop variables
	const ITYPE loop_dim = dim;
	ITYPE state_index;
#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// determin matrix pos
		UINT bit_val = (state_index >> target_qubit_index) % 2;

		// set value
		state[state_index] *= diagonal_matrix[bit_val];
	}
}
#endif
*/