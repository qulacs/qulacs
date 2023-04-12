
#include "MPIutil.hpp"
#include "update_ops.hpp"
#include "utility.hpp"

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void CZ_gate(UINT control_qubit_index, UINT target_qubit_index, CTYPE* state,
    ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    CZ_gate_parallel_simd(control_qubit_index, target_qubit_index, state, dim);
#elif defined(_USE_SVE)
    CZ_gate_parallel_sve(control_qubit_index, target_qubit_index, state, dim);
#else
    CZ_gate_parallel_unroll(
        control_qubit_index, target_qubit_index, state, dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void CZ_gate_parallel_unroll(UINT control_qubit_index, UINT target_qubit_index,
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

    const ITYPE mask = target_mask + control_mask;
    ITYPE state_index = 0;
    if (target_qubit_index == 0 || control_qubit_index == 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            state[basis_index] *= -1;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            state[basis_index] *= -1;
            state[basis_index + 1] *= -1;
        }
    }
}

#ifdef _USE_SIMD
void CZ_gate_parallel_simd(UINT control_qubit_index, UINT target_qubit_index,
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

    const ITYPE mask = target_mask + control_mask;
    ITYPE state_index = 0;
    if (target_qubit_index == 0 || control_qubit_index == 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            state[basis_index] *= -1;
        }
    } else {
        __m256d minus_one = _mm256_set_pd(-1, -1, -1, -1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            double* ptr = (double*)(state + basis_index);
            __m256d data = _mm256_loadu_pd(ptr);
            data = _mm256_mul_pd(data, minus_one);
            _mm256_storeu_pd(ptr, data);
        }
    }
}
#endif

#ifdef _USE_SVE
void CZ_gate_parallel_sve(UINT control_qubit_index, UINT target_qubit_index,
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

    const ITYPE mask = target_mask + control_mask;
    ITYPE state_index = 0;

    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;

    if (min_qubit_mask < VL) {
        CZ_gate_parallel_unroll(
            control_qubit_index, target_qubit_index, state, dim);
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += VL) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;

            svfloat64_t sv_data =
                svld1(svptrue_b64(), (double*)&state[basis_index]);
            sv_data = svneg_z(svptrue_b64(), sv_data);
            svst1(svptrue_b64(), (double*)&state[basis_index], sv_data);
        }
    }
}
#endif

#ifdef _USE_MPI
void CZ_gate_mpi(UINT control_qubit_index, UINT target_qubit_index,
    CTYPE* state, ITYPE dim, UINT inner_qc) {
    UINT left_qubit, right_qubit;
    if (control_qubit_index > target_qubit_index) {
        left_qubit = control_qubit_index;
        right_qubit = target_qubit_index;
    } else {
        left_qubit = target_qubit_index;
        right_qubit = control_qubit_index;
    }

    if (left_qubit < inner_qc) {
        CZ_gate(control_qubit_index, target_qubit_index, state, dim);
    } else if (right_qubit < inner_qc) {  // one quibit is outer
        const UINT rank = MPIutil::get_inst().get_rank();
        const UINT tgt_rank_bit = 1 << (left_qubit - inner_qc);
        if (rank & tgt_rank_bit) {
            Z_gate(right_qubit, state, dim);
        }     // if else, nothing to do.
    } else {  // both qubits are outer;
        const UINT rank = MPIutil::get_inst().get_rank();
        const UINT tgt0_rank_bit = 1 << (left_qubit - inner_qc);
        const UINT tgt1_rank_bit = 1 << (right_qubit - inner_qc);
        if (rank & tgt0_rank_bit && rank & tgt1_rank_bit) {
            state_multiply(-1., state, dim);
        }  // if else, nothing to do.
    }
}
#endif
