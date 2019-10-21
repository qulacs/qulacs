
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

void single_qubit_phase_gate_old_single(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim);
void single_qubit_phase_gate_old_parallel(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim);
void single_qubit_phase_gate_single(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim);
void single_qubit_phase_gate_unroll(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim);
void single_qubit_phase_gate_simd(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim);
void single_qubit_phase_gate_parallel(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim);

void single_qubit_phase_gate(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {
	//single_qubit_phase_gate_old_single(target_qubit_index, phase, state, dim);
	//single_qubit_phase_gate_old_parallel(target_qubit_index, phase, state, dim);
	//single_qubit_phase_gate_single(target_qubit_index, phase, state, dim);
	//single_qubit_phase_gate_unroll(target_qubit_index, phase, state, dim);
	//single_qubit_phase_gate_simd(target_qubit_index, phase, state, dim);
	//single_qubit_phase_gate_parallel(target_qubit_index, phase, state, dim);

#ifdef _OPENMP
	UINT threshold = 12;
	if (dim < (1ULL << threshold)) {
		single_qubit_phase_gate_simd(target_qubit_index, phase, state, dim);
	}
	else {
		single_qubit_phase_gate_parallel(target_qubit_index, phase, state, dim);
	}
#else
	single_qubit_phase_gate_simd(target_qubit_index, phase, state, dim);
#endif
}


void single_qubit_phase_gate_old_single(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {

	// target tmask
	const ITYPE mask = 1ULL << target_qubit_index;

	// loop varaibles
	const ITYPE loop_dim = dim / 2;
	ITYPE state_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {

		// crate index
		ITYPE basis_1 = insert_zero_to_basis_index(state_index, mask, target_qubit_index) ^ mask;

		// set values
		state[basis_1] *= phase;
	}
}


#ifdef _OPENMP
void single_qubit_phase_gate_old_parallel(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {

	// target tmask
	const ITYPE mask = 1ULL << target_qubit_index;

	// loop varaibles
	const ITYPE loop_dim = dim / 2;
	ITYPE state_index;

#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {

		// crate index
		ITYPE basis_1 = insert_zero_to_basis_index(state_index, mask, target_qubit_index) ^ mask;

		// set values
		state[basis_1] *= phase;
	}
}
#endif

void single_qubit_phase_gate_single(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {
	// target tmask
	const ITYPE mask = 1ULL << target_qubit_index;
	const ITYPE low_mask = mask - 1;
	const ITYPE high_mask = ~low_mask;

	// loop varaibles
	const ITYPE loop_dim = dim / 2;
	ITYPE state_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis = (state_index&low_mask) + ((state_index&high_mask)<<1) + mask;
		state[basis] *= phase;
	}
}



void single_qubit_phase_gate_unroll(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {
	// target tmask
	const ITYPE mask = 1ULL << target_qubit_index;
	const ITYPE low_mask = mask - 1;
	const ITYPE high_mask = ~low_mask;

	// loop varaibles
	const ITYPE loop_dim = dim / 2;
	if (target_qubit_index == 0) {
		ITYPE state_index;
		for (state_index = 1; state_index < dim; state_index+=2) {
			state[state_index] *= phase;
		}
	}
	else {
		ITYPE state_index;
		for (state_index = 0; state_index < loop_dim; state_index+=2) {
			ITYPE basis = (state_index&low_mask) + ((state_index&high_mask) << 1) + mask;
			state[basis] *= phase;
			state[basis+1] *= phase;
		}
	}
}


void single_qubit_phase_gate_simd(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {
	// target tmask
	const ITYPE mask = 1ULL << target_qubit_index;
	const ITYPE low_mask = mask - 1;
	const ITYPE high_mask = ~low_mask;

	// loop varaibles
	const ITYPE loop_dim = dim / 2;
	if (target_qubit_index == 0) {
		ITYPE state_index;
		for (state_index = 1; state_index < dim; state_index += 2) {
			state[state_index] *= phase;
		}
	}
	else {
		ITYPE state_index;
		__m256d mv0 = _mm256_set_pd(-cimag(phase), creal(phase), -cimag(phase), creal(phase));
		__m256d mv1 = _mm256_set_pd(creal(phase), cimag(phase), creal(phase), cimag(phase));
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis = (state_index&low_mask) + ((state_index&high_mask) << 1) + mask;
			double* ptr = (double*)(state + basis);
			__m256d data = _mm256_loadu_pd(ptr);
			__m256d data0 = _mm256_mul_pd(data, mv0);
			__m256d data1 = _mm256_mul_pd(data, mv1);
			data = _mm256_hadd_pd(data0, data1);
			_mm256_storeu_pd(ptr, data);
		}
	}
}

#ifdef _OPENMP
void single_qubit_phase_gate_parallel(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {
	// target tmask
	const ITYPE mask = 1ULL << target_qubit_index;
	const ITYPE low_mask = mask - 1;
	const ITYPE high_mask = ~low_mask;

	// loop varaibles
	const ITYPE loop_dim = dim / 2;
	if (target_qubit_index == 0) {
		ITYPE state_index;
#pragma omp parallel for
		for (state_index = 1; state_index < dim; state_index += 2) {
			state[state_index] *= phase;
		}
	}
	else {
		ITYPE state_index;
		__m256d mv0 = _mm256_set_pd(-cimag(phase), creal(phase), -cimag(phase), creal(phase));
		__m256d mv1 = _mm256_set_pd(creal(phase), cimag(phase), creal(phase), cimag(phase));
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis = (state_index&low_mask) + ((state_index&high_mask) << 1) + mask;
			double* ptr = (double*)(state + basis);
			__m256d data = _mm256_loadu_pd(ptr);
			__m256d data0 = _mm256_mul_pd(data, mv0);
			__m256d data1 = _mm256_mul_pd(data, mv1);
			data = _mm256_hadd_pd(data0, data1);
			_mm256_storeu_pd(ptr, data);
		}
	}
}
#endif
