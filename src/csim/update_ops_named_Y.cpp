
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

void Y_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    Y_gate_parallel_simd(target_qubit_index, state, dim);
#else
    Y_gate_parallel_unroll(target_qubit_index, state, dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void Y_gate_parallel_unroll(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    const CTYPE imag = 1.i;
    if (target_qubit_index == 0) {
        ITYPE basis_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (basis_index = 0; basis_index < dim; basis_index += 2) {
            CTYPE temp0 = state[basis_index];
            state[basis_index] = -imag * state[basis_index + 1];
            state[basis_index + 1] = imag * temp0;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
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

#ifdef _USE_SIMD
void Y_gate_parallel_simd(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    // const CTYPE imag = 1.i;
    __m256d minus_even = _mm256_set_pd(1, -1, 1, -1);
    __m256d minus_odd = _mm256_set_pd(-1, 1, -1, 1);
    __m256d minus_half = _mm256_set_pd(1, -1, -1, 1);
    if (target_qubit_index == 0) {
        ITYPE basis_index = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (basis_index = 0; basis_index < dim; basis_index += 2) {
            double* ptr0 = (double*)(state + basis_index);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            data0 = _mm256_permute4x64_pd(data0,
                27);  // (3210) -> (0123) : 16+4*2+3=27
            data0 = _mm256_mul_pd(data0, minus_half);
            _mm256_storeu_pd(ptr0, data0);
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_index_1 = basis_index_0 + mask;
            double* ptr0 = (double*)(state + basis_index_0);
            double* ptr1 = (double*)(state + basis_index_1);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            __m256d data1 = _mm256_loadu_pd(ptr1);
            data0 = _mm256_permute_pd(data0, 5);  // (3210) -> (2301) : 4+1
            data1 = _mm256_permute_pd(data1, 5);
            data0 = _mm256_mul_pd(data0, minus_even);
            data1 = _mm256_mul_pd(data1, minus_odd);
            _mm256_storeu_pd(ptr1, data0);
            _mm256_storeu_pd(ptr0, data1);
        }
    }
}
#endif
