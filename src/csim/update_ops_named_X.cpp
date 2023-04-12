
#include <cstring>

#include "MPIutil.hpp"
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

void X_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    X_gate_parallel_simd(target_qubit_index, state, dim);
#elif defined(_USE_SVE)
    X_gate_parallel_sve(target_qubit_index, state, dim);
#else
    X_gate_parallel_unroll(target_qubit_index, state, dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void X_gate_parallel_unroll(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
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
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
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

#ifdef _USE_SIMD
void X_gate_parallel_simd(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    // double* cast_state = (double*)state;
    if (target_qubit_index == 0) {
        ITYPE basis_index = 0;
#pragma omp parallel for
        for (basis_index = 0; basis_index < dim; basis_index += 2) {
            double* ptr = (double*)(state + basis_index);
            __m256d data = _mm256_loadu_pd(ptr);
            data = _mm256_permute4x64_pd(data,
                78);  // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
            _mm256_storeu_pd(ptr, data);
        }
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
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

#ifdef _USE_SVE
void X_gate_parallel_sve(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;

    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;

    if (mask < VL) {
        X_gate_parallel_unroll(target_qubit_index, state, dim);
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += VL) {
            ITYPE basis_index_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_index_1 = basis_index_0 + mask;

            svfloat64_t sv_data0 =
                svld1(svptrue_b64(), (double*)&state[basis_index_0]);
            svfloat64_t sv_data1 =
                svld1(svptrue_b64(), (double*)&state[basis_index_1]);
            svst1(svptrue_b64(), (double*)&state[basis_index_0], sv_data1);
            svst1(svptrue_b64(), (double*)&state[basis_index_1], sv_data0);
        }
    }
}
#endif

#ifdef _USE_MPI
void X_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        X_gate(target_qubit_index, state, dim);
    } else {
        MPIutil& m = MPIutil::get_inst();
        const int rank = m.get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* t = m.get_workarea(&dim_work, &num_work);
        assert(num_work > 0);
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        const int pair_rank = rank ^ pair_rank_bit;
        CTYPE* si = state;
        for (ITYPE i = 0; i < num_work; ++i) {
            m.m_DC_sendrecv(si, t, dim_work, pair_rank);
            memcpy(si, t, dim_work * sizeof(CTYPE));
            si += dim_work;
        }
    }
}
#endif
