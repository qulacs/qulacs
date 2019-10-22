
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

//void X_gate_old(UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void X_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void X_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim);

void X_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	//UINT threshold = 13;
	//X_gate_old(target_qubit_index, state, dim);
	//X_gate_single(target_qubit_index, state, dim);
	//X_gate_single_simd(target_qubit_index, state, dim);
	//X_gate_single_unroll(target_qubit_index, state, dim);
	//X_gate_parallel(target_qubit_index, state, dim);
	//return;

#ifdef _USE_SIMD
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		X_gate_single_simd(target_qubit_index, state, dim);
	}
	else {
		X_gate_parallel_simd(target_qubit_index, state, dim);
	}
#else
	X_gate_single_simd(target_qubit_index, state, dim);
#endif
#else
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		X_gate_single_unroll(target_qubit_index, state, dim);
	}
	else {
		X_gate_parallel_unroll(target_qubit_index, state, dim);
	}
#else
	X_gate_single_unroll(target_qubit_index, state, dim);
#endif
#endif
}


void X_gate_single_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	if (target_qubit_index == 0) {
		ITYPE basis_index = 0;
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			CTYPE temp = state[basis_index];
			state[basis_index] = state[basis_index + 1];
			state[basis_index + 1] = temp;
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

#ifdef _OPENMP
void X_gate_parallel_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	if (target_qubit_index == 0) {
		ITYPE basis_index = 0;
#pragma omp parallel for
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			CTYPE temp = state[basis_index];
			state[basis_index] = state[basis_index + 1];
			state[basis_index + 1] = temp;
		}
	}
	else {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			CTYPE temp0 = state[basis_index_0];
			CTYPE temp1 = state[basis_index_0 + 1];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_0 + 1] = state[basis_index_1 + 1];
			state[basis_index_1] = temp0;
			state[basis_index_1 + 1] = temp1;
		}
	}
}
#endif

#ifdef _USE_SIMD
void X_gate_single_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	//double* cast_state = (double*)state;
	if (target_qubit_index == 0) {
		ITYPE basis_index = 0;
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			double* ptr = (double*)(state + basis_index);
			__m256d data = _mm256_loadu_pd(ptr);
			data = _mm256_permute4x64_pd(data, 78); // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
			_mm256_storeu_pd(ptr, data);
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
void X_gate_parallel_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	//double* cast_state = (double*)state;
	if (target_qubit_index == 0) {
		ITYPE basis_index = 0;
#pragma omp parallel for
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			double* ptr = (double*)(state + basis_index);
			__m256d data = _mm256_loadu_pd(ptr);
			data = _mm256_permute4x64_pd(data, 78); // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
			_mm256_storeu_pd(ptr, data);
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
			_mm256_storeu_pd(ptr1, data0);
			_mm256_storeu_pd(ptr0, data1);
		}
	}
}
#endif
#endif


/*


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

*/