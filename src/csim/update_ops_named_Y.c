
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

//void Y_gate_old_single(UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void Y_gate_old_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void Y_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void Y_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim);

void Y_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	//Y_gate_old_single(target_qubit_index, state, dim);
	//Y_gate_old_parallel(target_qubit_index, state, dim);
	//Y_gate_single(target_qubit_index, state, dim);
	//Y_gate_single_simd(target_qubit_index, state, dim);
	//Y_gate_single_unroll(target_qubit_index, state, dim);
	//Y_gate_parallel(target_qubit_index, state, dim);
	//return;

#ifdef _USE_SIMD
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		Y_gate_single_simd(target_qubit_index, state, dim);
	}
	else {
		Y_gate_parallel_simd(target_qubit_index, state, dim);
	}
#else
	Y_gate_single_simd(target_qubit_index, state, dim);
#endif
#else
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		Y_gate_single_unroll(target_qubit_index, state, dim);
	}
	else {
		Y_gate_parallel_unroll(target_qubit_index, state, dim);
	}
#else
	Y_gate_single_unroll(target_qubit_index, state, dim);
#endif
#endif
}

void Y_gate_single_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	const CTYPE imag = 1.i;
	if (target_qubit_index == 0) {
		ITYPE basis_index;
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			CTYPE temp0 = state[basis_index];
			state[basis_index] = -imag * state[basis_index + 1];
			state[basis_index + 1] = imag * temp0;
		}
	} else {
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			CTYPE temp0 = state[basis_index_0];
			CTYPE temp1 = state[basis_index_0+1];
			state[basis_index_0] = -imag * state[basis_index_1];
			state[basis_index_0+1] = -imag * state[basis_index_1+1];
			state[basis_index_1] = imag * temp0;
			state[basis_index_1+1] = imag * temp1;
		}
	}
}

#ifdef _OPENMP
void Y_gate_parallel_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	const CTYPE imag = 1.i;
	if (target_qubit_index == 0) {
		ITYPE basis_index;
#pragma omp parallel for 
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			CTYPE temp0 = state[basis_index];
			state[basis_index] = -imag * state[basis_index + 1];
			state[basis_index + 1] = imag * temp0;
		}
	}
	else {
#pragma omp parallel for 
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			CTYPE temp0 = state[basis_index_0];
			CTYPE temp1 = state[basis_index_0 + 1];
			state[basis_index_0] = -imag * state[basis_index_1];
			state[basis_index_0 + 1] = -imag * state[basis_index_1 + 1];
			state[basis_index_1] = imag * temp0;
			state[basis_index_1 + 1] = imag * temp1;
		}
	}
}
#endif

#ifdef _USE_SIMD
void Y_gate_single_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	//const CTYPE imag = 1.i;
	__m256d minus_even = _mm256_set_pd(1, -1, 1, -1);
	__m256d minus_odd = _mm256_set_pd(-1, 1, -1, 1);
	__m256d minus_half = _mm256_set_pd(1, -1, -1, 1);
	if (target_qubit_index == 0) {
		ITYPE basis_index = 0;
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			double* ptr0 = (double*)(state + basis_index);
			__m256d data0 = _mm256_loadu_pd(ptr0);
			data0 = _mm256_permute4x64_pd(data0, 27); // (3210) -> (0123) : 16+4*2+3=27
			data0 = _mm256_mul_pd(data0, minus_half);
			_mm256_storeu_pd(ptr0, data0);
		}
	}
	else {
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			double* ptr0 = (double*)(state + basis_index_0);
			double* ptr1 = (double*)(state + basis_index_1);
			__m256d data0 = _mm256_loadu_pd(ptr0);
			__m256d data1 = _mm256_loadu_pd(ptr1);
			data0 = _mm256_permute_pd(data0, 5); // (3210) -> (2301) : 4+1
			data1 = _mm256_permute_pd(data1, 5);
			data0 = _mm256_mul_pd(data0, minus_even);
			data1 = _mm256_mul_pd(data1, minus_odd);
			_mm256_storeu_pd(ptr1, data0);
			_mm256_storeu_pd(ptr0, data1);
		}
	}
}

#ifdef _OPENMP
void Y_gate_parallel_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	//const CTYPE imag = 1.i;
	__m256d minus_even = _mm256_set_pd(1, -1, 1, -1);
	__m256d minus_odd = _mm256_set_pd(-1, 1, -1, 1);
	__m256d minus_half = _mm256_set_pd(1, -1, -1, 1);
	if (target_qubit_index == 0) {
		ITYPE basis_index = 0;
#pragma omp parallel for
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			double* ptr0 = (double*)(state + basis_index);
			__m256d data0 = _mm256_loadu_pd(ptr0);
			data0 = _mm256_permute4x64_pd(data0, 27); // (3210) -> (0123) : 16+4*2+3=27
			data0 = _mm256_mul_pd(data0, minus_half);
			_mm256_storeu_pd(ptr0, data0);
		}
	}
	else {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			double* ptr0 = (double*)(state + basis_index_0);
			double* ptr1 = (double*)(state + basis_index_1);
			__m256d data0 = _mm256_loadu_pd(ptr0);
			__m256d data1 = _mm256_loadu_pd(ptr1);
			data0 = _mm256_permute_pd(data0, 5); // (3210) -> (2301) : 4+1
			data1 = _mm256_permute_pd(data1, 5);
			data0 = _mm256_mul_pd(data0, minus_even);
			data1 = _mm256_mul_pd(data1, minus_odd);
			_mm256_storeu_pd(ptr1, data0);
			_mm256_storeu_pd(ptr0, data1);
		}
	}
}
#endif

#endif



/*
#ifdef _OPENMP
void Y_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	const CTYPE imag = 1.i;
#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
		ITYPE basis_index_1 = basis_index_0 + mask;
		CTYPE temp = state[basis_index_0];
		state[basis_index_0] = -imag * state[basis_index_1];
		state[basis_index_0] = imag * temp;
	}
}
#endif

void Y_gate_old_single(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	ITYPE state_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = insert_zero_to_basis_index(state_index, mask, target_qubit_index);
		ITYPE basis_index_1 = basis_index_0 ^ mask;
		CTYPE cval_0 = state[basis_index_0];
		CTYPE cval_1 = state[basis_index_1];
		state[basis_index_0] = -cval_1 * 1.i;
		state[basis_index_1] = cval_0 * 1.i;
	}
}

void Y_gate_old_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = insert_zero_to_basis_index(state_index, mask, target_qubit_index);
		ITYPE basis_index_1 = basis_index_0 ^ mask;
		CTYPE cval_0 = state[basis_index_0];
		CTYPE cval_1 = state[basis_index_1];
		state[basis_index_0] = -cval_1 * 1.i;
		state[basis_index_1] = cval_0 * 1.i;
	}
}

void Y_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	const CTYPE imag = 1.i;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
		ITYPE basis_index_1 = basis_index_0 + mask;
		CTYPE temp = state[basis_index_0];
		state[basis_index_0] = - imag * state[basis_index_1];
		state[basis_index_0] = imag * temp;
	}
}

*/