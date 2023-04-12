
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

void Z_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    Z_gate_parallel_simd(target_qubit_index, state, dim);
#elif defined(_USE_SVE)
    Z_gate_parallel_sve(target_qubit_index, state, dim);
#else
    Z_gate_parallel_unroll(target_qubit_index, state, dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void Z_gate_parallel_unroll(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    if (target_qubit_index == 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 1; state_index < dim; state_index += 2) {
            state[state_index] *= -1;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & mask_low) +
                                ((state_index & mask_high) << 1) + mask;
            state[basis_index] *= -1;
            state[basis_index + 1] *= -1;
        }
    }
}

#ifdef _USE_SIMD
void Z_gate_parallel_simd(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    __m256d minus_one = _mm256_set_pd(-1, -1, -1, -1);
    if (target_qubit_index == 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 1; state_index < dim; state_index += 2) {
            state[state_index] *= -1;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & mask_low) +
                                ((state_index & mask_high) << 1) + mask;
            double* ptr0 = (double*)(state + basis_index);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            data0 = _mm256_mul_pd(data0, minus_one);
            _mm256_storeu_pd(ptr0, data0);
        }
    }
}
#endif

#ifdef _USE_SVE
void Z_gate_parallel_sve(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;

    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;

    if (mask < VL) {
        Z_gate_parallel_unroll(target_qubit_index, state, dim);
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += VL) {
            ITYPE basis_index = (state_index & mask_low) +
                                ((state_index & mask_high) << 1) + mask;
            svfloat64_t sv_data =
                svld1(svptrue_b64(), (double*)&state[basis_index]);
            sv_data = svneg_z(svptrue_b64(), sv_data);
            svst1(svptrue_b64(), (double*)&state[basis_index], sv_data);
        }
    }
}
#endif

#ifdef _USE_MPI
void Z_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        Z_gate(target_qubit_index, state, dim);
    } else {
        const int rank = MPIutil::get_inst().get_rank();
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        if (rank & pair_rank_bit) {
            state_multiply(-1., state, dim);
        }
    }
}
#endif
