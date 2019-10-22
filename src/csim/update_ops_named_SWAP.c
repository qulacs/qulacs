
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

//void SWAP_gate_old_single(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim);
//void SWAP_gate_old_parallel(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim);
//void SWAP_gate_single(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim);
//void SWAP_gate_parallel(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim);

void SWAP_gate(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim) {
	//SWAP_gate_old_single(target_qubit_index_0, target_qubit_index_1, state, dim);
	//SWAP_gate_old_parallel(target_qubit_index_0, target_qubit_index_1, state, dim);
	//SWAP_gate_single(target_qubit_index_0, target_qubit_index_1, state, dim);
	//SWAP_gate_single_unroll(target_qubit_index_0, target_qubit_index_1, state, dim);
	//SWAP_gate_single_simd(target_qubit_index_0, target_qubit_index_1, state, dim);
	//SWAP_gate_parallel(target_qubit_index_0, target_qubit_index_1, state, dim);
	//return;

#ifdef _USE_SIMD
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		SWAP_gate_single_simd(target_qubit_index_0, target_qubit_index_1, state, dim);
	}
	else {
		SWAP_gate_parallel_simd(target_qubit_index_0, target_qubit_index_1, state, dim);
	}
#else
	SWAP_gate_single_simd(target_qubit_index_0, target_qubit_index_1, state, dim);
#endif
#else
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		SWAP_gate_single_unroll(target_qubit_index_0, target_qubit_index_1, state, dim);
	}
	else {
		SWAP_gate_parallel_unroll(target_qubit_index_0, target_qubit_index_1, state, dim);
	}
#else
	SWAP_gate_single_unroll(target_qubit_index_0, target_qubit_index_1, state, dim);
#endif
#endif
}


void SWAP_gate_single_unroll(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE mask_0 = 1ULL << target_qubit_index_0;
	const ITYPE mask_1 = 1ULL << target_qubit_index_1;
	const ITYPE mask = mask_0 + mask_1;

	const UINT min_qubit_index = get_min_ui(target_qubit_index_0, target_qubit_index_1);
	const UINT max_qubit_index = get_max_ui(target_qubit_index_0, target_qubit_index_1);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index = 0;
	if (target_qubit_index_0 == 0 || target_qubit_index_1 == 0) {
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ mask_0;
			ITYPE basis_index_1 = basis_index_0 ^ mask;
			CTYPE temp = state[basis_index_0];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_1] = temp;
		}
	}else{
		// a,a+1 is swapped to a^m, a^m+1, respectively
		for (state_index = 0; state_index < loop_dim; state_index+=2) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ mask_0;
			ITYPE basis_index_1 = basis_index_0 ^ mask;
			CTYPE temp0 = state[basis_index_0];
			CTYPE temp1 = state[basis_index_0 + 1];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_0 + 1] = state[basis_index_1 + 1];
			state[basis_index_1] = temp0;
			state[basis_index_1 + 1] = temp1;
		}
	}
}

#ifdef _OPENMP
void SWAP_gate_parallel_unroll(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE mask_0 = 1ULL << target_qubit_index_0;
	const ITYPE mask_1 = 1ULL << target_qubit_index_1;
	const ITYPE mask = mask_0 + mask_1;

	const UINT min_qubit_index = get_min_ui(target_qubit_index_0, target_qubit_index_1);
	const UINT max_qubit_index = get_max_ui(target_qubit_index_0, target_qubit_index_1);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index = 0;
	if (target_qubit_index_0 == 0 || target_qubit_index_1 == 0) {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ mask_0;
			ITYPE basis_index_1 = basis_index_0 ^ mask;
			CTYPE temp = state[basis_index_0];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_1] = temp;
		}
	}
	else {
		// a,a+1 is swapped to a^m, a^m+1, respectively
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ mask_0;
			ITYPE basis_index_1 = basis_index_0 ^ mask;
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
void SWAP_gate_single_simd(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE mask_0 = 1ULL << target_qubit_index_0;
	const ITYPE mask_1 = 1ULL << target_qubit_index_1;
	const ITYPE mask = mask_0 + mask_1;

	const UINT min_qubit_index = get_min_ui(target_qubit_index_0, target_qubit_index_1);
	const UINT max_qubit_index = get_max_ui(target_qubit_index_0, target_qubit_index_1);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index = 0;
	if (target_qubit_index_0 == 0 || target_qubit_index_1 == 0) {
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ mask_0;
			ITYPE basis_index_1 = basis_index_0 ^ mask;
			CTYPE temp = state[basis_index_0];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_1] = temp;
		}
	}
	else {
		// a,a+1 is swapped to a^m, a^m+1, respectively
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ mask_0;
			ITYPE basis_index_1 = basis_index_0 ^ mask;
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
void SWAP_gate_parallel_simd(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE mask_0 = 1ULL << target_qubit_index_0;
	const ITYPE mask_1 = 1ULL << target_qubit_index_1;
	const ITYPE mask = mask_0 + mask_1;

	const UINT min_qubit_index = get_min_ui(target_qubit_index_0, target_qubit_index_1);
	const UINT max_qubit_index = get_max_ui(target_qubit_index_0, target_qubit_index_1);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index = 0;
	if (target_qubit_index_0 == 0 || target_qubit_index_1 == 0) {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ mask_0;
			ITYPE basis_index_1 = basis_index_0 ^ mask;
			CTYPE temp = state[basis_index_0];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_1] = temp;
		}
	}
	else {
		// a,a+1 is swapped to a^m, a^m+1, respectively
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ mask_0;
			ITYPE basis_index_1 = basis_index_0 ^ mask;
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
#ifdef _OPENMP
void SWAP_gate_parallel(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE mask_0 = 1ULL << target_qubit_index_0;
	const ITYPE mask_1 = 1ULL << target_qubit_index_1;
	const ITYPE mask = mask_0 + mask_1;

	const UINT min_qubit_index = get_min_ui(target_qubit_index_0, target_qubit_index_1);
	const UINT max_qubit_index = get_max_ui(target_qubit_index_0, target_qubit_index_1);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index = 0;
#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&low_mask)
			+ ((state_index&mid_mask) << 1)
			+ ((state_index&high_mask) << 2)
			+ mask_0;
		ITYPE basis_index_1 = basis_index_0 ^ mask;
		CTYPE temp = state[basis_index_0];
		state[basis_index_0] = state[basis_index_1];
		state[basis_index_1] = temp;
	}
}
#endif


void SWAP_gate_old_single(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;
	const UINT min_qubit_index = get_min_ui(target_qubit_index_0, target_qubit_index_1);
	const UINT max_qubit_index = get_max_ui(target_qubit_index_0, target_qubit_index_1);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << max_qubit_index;
	const ITYPE target_mask_0 = 1ULL << target_qubit_index_0;
	const ITYPE target_mask_1 = 1ULL << target_qubit_index_1;
	ITYPE state_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_insert_only_min = insert_zero_to_basis_index(state_index, min_qubit_mask, min_qubit_index);
		ITYPE basis_00 = insert_zero_to_basis_index(basis_insert_only_min, max_qubit_mask, max_qubit_index);
		ITYPE basis_01 = basis_00 ^ target_mask_0;
		ITYPE basis_10 = basis_00 ^ target_mask_1;
		swap_amplitude(state, basis_01, basis_10);
	}
}

#ifdef _OPENMP
void SWAP_gate_old_parallel(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;
	const UINT min_qubit_index = get_min_ui(target_qubit_index_0, target_qubit_index_1);
	const UINT max_qubit_index = get_max_ui(target_qubit_index_0, target_qubit_index_1);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << max_qubit_index;
	const ITYPE target_mask_0 = 1ULL << target_qubit_index_0;
	const ITYPE target_mask_1 = 1ULL << target_qubit_index_1;
	ITYPE state_index;
#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_insert_only_min = insert_zero_to_basis_index(state_index, min_qubit_mask, min_qubit_index);
		ITYPE basis_00 = insert_zero_to_basis_index(basis_insert_only_min, max_qubit_mask, max_qubit_index);
		ITYPE basis_01 = basis_00 ^ target_mask_0;
		ITYPE basis_10 = basis_00 ^ target_mask_1;
		swap_amplitude(state, basis_01, basis_10);
	}
}
#endif


void SWAP_gate_single(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE mask_0 = 1ULL << target_qubit_index_0;
	const ITYPE mask_1 = 1ULL << target_qubit_index_1;
	const ITYPE mask = mask_0 + mask_1;

	const UINT min_qubit_index = get_min_ui(target_qubit_index_0, target_qubit_index_1);
	const UINT max_qubit_index = get_max_ui(target_qubit_index_0, target_qubit_index_1);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index-1);
	const ITYPE low_mask = min_qubit_mask-1;
	const ITYPE mid_mask = (max_qubit_mask-1)^low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index = 0;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&low_mask)
			+ ((state_index&mid_mask) << 1)
			+ ((state_index&high_mask) << 2)
			+ mask_0;
		ITYPE basis_index_1 = basis_index_0 ^ mask;
		CTYPE temp = state[basis_index_0];
		state[basis_index_0] = state[basis_index_1];
		state[basis_index_1] = temp;
	}
}
*/