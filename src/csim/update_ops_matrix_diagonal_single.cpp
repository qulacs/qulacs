
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.hpp"
#include "update_ops.hpp"
#include "utility.hpp"
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

void single_qubit_diagonal_matrix_gate(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 12);
#endif

#ifdef _USE_SIMD
    single_qubit_diagonal_matrix_gate_parallel_simd(
        target_qubit_index, diagonal_matrix, state, dim);
#else
    single_qubit_diagonal_matrix_gate_parallel_unroll(
        target_qubit_index, diagonal_matrix, state, dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void single_qubit_diagonal_matrix_gate_parallel_unroll(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE* state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    if (target_qubit_index == 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            state[state_index] *= diagonal_matrix[0];
            state[state_index + 1] *= diagonal_matrix[1];
        }
    } else {
        ITYPE mask = 1ULL << target_qubit_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            int bitval = ((state_index & mask) != 0);
            state[state_index] *= diagonal_matrix[bitval];
            state[state_index + 1] *= diagonal_matrix[bitval];
        }
    }
}

#ifdef _USE_SIMD
void single_qubit_diagonal_matrix_gate_parallel_simd(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE* state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    if (target_qubit_index == 0) {
        __m256d mv0 = _mm256_set_pd(-_cimag(diagonal_matrix[1]),
            _creal(diagonal_matrix[1]), -_cimag(diagonal_matrix[0]),
            _creal(diagonal_matrix[0]));
        __m256d mv1 = _mm256_set_pd(_creal(diagonal_matrix[1]),
            _cimag(diagonal_matrix[1]), _creal(diagonal_matrix[0]),
            _cimag(diagonal_matrix[0]));
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            double* ptr = (double*)(state + state_index);
            __m256d data = _mm256_loadu_pd(ptr);
            __m256d data0 = _mm256_mul_pd(data, mv0);
            __m256d data1 = _mm256_mul_pd(data, mv1);
            data = _mm256_hadd_pd(data0, data1);
            _mm256_storeu_pd(ptr, data);
        }
    } else {
        __m256d mv0 = _mm256_set_pd(-_cimag(diagonal_matrix[0]),
            _creal(diagonal_matrix[0]), -_cimag(diagonal_matrix[0]),
            _creal(diagonal_matrix[0]));
        __m256d mv1 = _mm256_set_pd(_creal(diagonal_matrix[0]),
            _cimag(diagonal_matrix[0]), _creal(diagonal_matrix[0]),
            _cimag(diagonal_matrix[0]));
        __m256d mv2 = _mm256_set_pd(-_cimag(diagonal_matrix[1]),
            _creal(diagonal_matrix[1]), -_cimag(diagonal_matrix[1]),
            _creal(diagonal_matrix[1]));
        __m256d mv3 = _mm256_set_pd(_creal(diagonal_matrix[1]),
            _cimag(diagonal_matrix[1]), _creal(diagonal_matrix[1]),
            _cimag(diagonal_matrix[1]));
        //__m256i mask = _mm256_set1_epi64x(1LL<<target_qubit_index);
        ITYPE mask = 1LL << target_qubit_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
