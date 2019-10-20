
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

void X_gate_old(UINT target_qubit_index, CTYPE *state, ITYPE dim);
void X_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim);
void X_gate_single_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim);
void X_gate_single_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim);
void X_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim);
void X_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
#ifdef _OPENMP
	//UINT threshold = 13;
	//X_gate_old(target_qubit_index, state, dim);
	//X_gate_single(target_qubit_index, state, dim);
	//X_gate_single_simd(target_qubit_index, state, dim);
	//X_gate_single_unroll(target_qubit_index, state, dim);
	//X_gate_parallel(target_qubit_index, state, dim);
	//return;
	UINT threshold = 13;
	if (dim < (1ULL << threshold)) {
		X_gate_single_unroll(target_qubit_index, state, dim);
	}
	else {
		X_gate_parallel(target_qubit_index, state, dim);
	}
#else
	X_gate_single_unroll(target_qubit_index, state, dim);
#endif
}


void X_gate_old(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	ITYPE state_index;
#ifdef _OPENMP
//#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = insert_zero_to_basis_index(state_index, mask, target_qubit_index);
		ITYPE basis_index_1 = basis_index_0 ^ mask;
		swap_amplitude(state, basis_index_0, basis_index_1);
	}
}

void X_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
		ITYPE basis_index_1 = basis_index_0 + mask;
		CTYPE temp = state[basis_index_0];
		state[basis_index_0] = state[basis_index_1];
		state[basis_index_1] = temp;
	}
}

void X_gate_single_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	if (target_qubit_index == 0) {
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index = (state_index&mask_low) + ((state_index&mask_high) << 1);
			CTYPE temp = state[basis_index];
			state[basis_index] = state[basis_index + 1];
			state[basis_index + 1] = state[basis_index];
		}
	}
	else {
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			CTYPE temp0 = state[basis_index_0];
			CTYPE temp1 = state[basis_index_0+1];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_0+1] = state[basis_index_1+1];
			state[basis_index_1] = temp0;
			state[basis_index_1+1] = temp1;
		}
	}
}

void X_gate_single_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	double* cast_state = (double*)state;
	if (target_qubit_index == 0) {
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index = ((state_index&mask_low) + ((state_index&mask_high) << 1))<<1;
			__m256d data = _mm256_loadu_pd(cast_state+basis_index);
			data = _mm256_permute4x64_pd(data, 78); // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
			_mm256_storeu_pd(cast_state+basis_index, data);
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
			_mm256_storeu_pd(ptr1, data0);
			_mm256_storeu_pd(ptr0, data1);
		}
	}
}

#ifdef _OPENMP
void X_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
		ITYPE basis_index_1 = basis_index_0 + mask;
		CTYPE temp = state[basis_index_0];
		state[basis_index_0] = state[basis_index_1];
		state[basis_index_1] = temp;
	}
}
#endif