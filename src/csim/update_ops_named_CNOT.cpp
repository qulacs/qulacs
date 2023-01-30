#include <stddef.h>

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

void CNOT_gate(UINT control_qubit_index, UINT target_qubit_index, CTYPE* state,
    ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    CNOT_gate_parallel_simd(
        control_qubit_index, target_qubit_index, state, dim);
#else
    CNOT_gate_parallel_unroll(
        control_qubit_index, target_qubit_index, state, dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void CNOT_gate_parallel_unroll(UINT control_qubit_index,
    UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;

    const UINT min_qubit_index =
        get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index =
        get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index = 0;
    if (target_qubit_index == 0) {
        // swap neighboring two basis
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + control_mask;
            CTYPE temp = state[basis_index];
            state[basis_index] = state[basis_index + 1];
            state[basis_index + 1] = temp;
        }
    } else if (control_qubit_index == 0) {
        // no neighboring swap
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
            ITYPE basis_index_1 = basis_index_0 + target_mask;
            CTYPE temp = state[basis_index_0];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_1] = temp;
        }
    } else {
        // a,a+1 is swapped to a^m, a^m+1, respectively
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
            ITYPE basis_index_1 = basis_index_0 + target_mask;
            CTYPE temp0 = state[basis_index_0];
            CTYPE temp1 = state[basis_index_0 + 1];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_0 + 1] = state[basis_index_1 + 1];
            state[basis_index_1] = temp0;
            state[basis_index_1 + 1] = temp1;
        }
    }
}

#ifdef _USE_SIMD
void CNOT_gate_parallel_simd(UINT control_qubit_index, UINT target_qubit_index,
    CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;

    const UINT min_qubit_index =
        get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index =
        get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index = 0;
    if (target_qubit_index == 0) {
        // swap neighboring two basis
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + control_mask;
            double* ptr = (double*)(state + basis_index);
            __m256d data = _mm256_loadu_pd(ptr);
            data = _mm256_permute4x64_pd(data,
                78);  // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
            _mm256_storeu_pd(ptr, data);
        }
    } else if (control_qubit_index == 0) {
        // no neighboring swap
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
            ITYPE basis_index_1 = basis_index_0 + target_mask;
            CTYPE temp = state[basis_index_0];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_1] = temp;
        }
    } else {
        // a,a+1 is swapped to a^m, a^m+1, respectively
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
            ITYPE basis_index_1 = basis_index_0 + target_mask;
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
