
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

//void Z_gate_old_single(UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void Z_gate_old_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void Z_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void Z_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim);

void Z_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	//Z_gate_old_single(target_qubit_index, state, dim);
	//Z_gate_old_parallel(target_qubit_index, state, dim);
	//Z_gate_single(target_qubit_index, state, dim);
	//Z_gate_single_simd(target_qubit_index, state, dim);
	//Z_gate_single_unroll(target_qubit_index, state, dim);
	//Z_gate_parallel(target_qubit_index, state, dim);
	//return;

#ifdef _USE_SIMD
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		Z_gate_single_simd(target_qubit_index, state, dim);
	}
	else {
		Z_gate_parallel_simd(target_qubit_index, state, dim);
	}
#else
	Z_gate_single_simd(target_qubit_index, state, dim);
#endif
#else
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		Z_gate_single_unroll(target_qubit_index, state, dim);
	}
	else {
		Z_gate_parallel_unroll(target_qubit_index, state, dim);
	}
#else
	Z_gate_single_unroll(target_qubit_index, state, dim);
#endif
#endif
}

void Z_gate_single_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	if (target_qubit_index == 0) {
		for (state_index = 1; state_index < dim; state_index += 2) {
			state[state_index] *= -1;
		}
	}
	else {
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index = (state_index&mask_low) + ((state_index&mask_high) << 1) + mask;
			state[basis_index] *= -1;
			state[basis_index+1] *= -1;
		}
	}
}

#ifdef _OPENMP
void Z_gate_parallel_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	if (target_qubit_index == 0) {
#pragma omp parallel for
		for (state_index = 1; state_index < dim; state_index += 2) {
			state[state_index] *= -1;
		}
	}
	else {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index = (state_index&mask_low) + ((state_index&mask_high) << 1) + mask;
			state[basis_index] *= -1;
			state[basis_index + 1] *= -1;
		}
	}
}
#endif

#ifdef _USE_SIMD
void Z_gate_single_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	__m256d minus_one = _mm256_set_pd(-1,-1,-1,-1);
	if (target_qubit_index == 0) {
		for (state_index = 1; state_index < dim; state_index += 2) {
			state[state_index] *= -1;
		}
	}
	else {
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index = (state_index&mask_low) + ((state_index&mask_high) << 1) + mask;
			double* ptr0 = (double*)(state + basis_index);
			__m256d data0 = _mm256_loadu_pd(ptr0);
			data0 = _mm256_mul_pd(data0, minus_one);
			_mm256_storeu_pd(ptr0, data0);
		}
	}
}

#ifdef _OPENMP
void Z_gate_parallel_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	__m256d minus_one = _mm256_set_pd(-1, -1, -1, -1);
	if (target_qubit_index == 0) {
#pragma omp parallel for
		for (state_index = 1; state_index < dim; state_index += 2) {
			state[state_index] *= -1;
		}
	}
	else {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index = (state_index&mask_low) + ((state_index&mask_high) << 1) + mask;
			double* ptr0 = (double*)(state + basis_index);
			__m256d data0 = _mm256_loadu_pd(ptr0);
			data0 = _mm256_mul_pd(data0, minus_one);
			_mm256_storeu_pd(ptr0, data0);
		}
	}
}
#endif

#endif





/*


void Z_gate_old_single(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	ITYPE state_index;
	ITYPE mask = (1ULL << target_qubit_index);
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE temp_index = insert_zero_to_basis_index(state_index, mask, target_qubit_index) ^ mask;
		state[temp_index] *= -1;
	}
}

void Z_gate_old_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	ITYPE state_index;
	ITYPE mask = (1ULL << target_qubit_index);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE temp_index = insert_zero_to_basis_index(state_index, mask, target_qubit_index) ^ mask;
		state[temp_index] *= -1;
	}
}

void Z_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index = (state_index&mask_low) + ((state_index&mask_high) << 1) + mask;
		state[basis_index] *= -1;
	}
}

#ifdef _OPENMP
void Z_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index = (state_index&mask_low) + ((state_index&mask_high) << 1) + mask;
		state[basis_index] *= -1;
	}
}
#endif

*/