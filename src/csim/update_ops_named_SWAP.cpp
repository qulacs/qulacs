
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

void SWAP_gate(UINT target_qubit_index_0, UINT target_qubit_index_1,
    CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    SWAP_gate_parallel_simd(
        target_qubit_index_0, target_qubit_index_1, state, dim);
#elif defined(_USE_SVE)
    SWAP_gate_parallel_sve(
        target_qubit_index_0, target_qubit_index_1, state, dim);
#else
    SWAP_gate_parallel_unroll(
        target_qubit_index_0, target_qubit_index_1, state, dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void SWAP_gate_parallel_unroll(UINT target_qubit_index_0,
    UINT target_qubit_index_1, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE mask_1 = 1ULL << target_qubit_index_1;
    const ITYPE mask = mask_0 + mask_1;

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index_0, target_qubit_index_1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index = 0;
    if (target_qubit_index_0 == 0 || target_qubit_index_1 == 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;
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
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
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

#ifdef _USE_SIMD
void SWAP_gate_parallel_simd(UINT target_qubit_index_0,
    UINT target_qubit_index_1, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE mask_1 = 1ULL << target_qubit_index_1;
    const ITYPE mask = mask_0 + mask_1;

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index_0, target_qubit_index_1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index = 0;
    if (target_qubit_index_0 == 0 || target_qubit_index_1 == 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;
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
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
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

#ifdef _USE_SVE
void SWAP_gate_parallel_sve(UINT target_qubit_index_0,
    UINT target_qubit_index_1, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE mask_1 = 1ULL << target_qubit_index_1;
    const ITYPE mask = mask_0 + mask_1;

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index_0, target_qubit_index_1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index = 0;

    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;

    if ((dim > VL) && (min_qubit_mask >= VL)) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += VL) {
            // Calculate indices
            ITYPE basis_0 = (state_index & low_mask) +
                            ((state_index & mid_mask) << 1) +
                            ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_1 = basis_0 ^ mask;

            // Load values
            svfloat64_t input0 = svld1(svptrue_b64(), (double*)&state[basis_0]);
            svfloat64_t input1 = svld1(svptrue_b64(), (double*)&state[basis_1]);

            // Store values
            svst1(svptrue_b64(), (double*)&state[basis_0], input1);
            svst1(svptrue_b64(), (double*)&state[basis_1], input0);
        }
    } else {  // if ((dim > VL) && (min_qubit_mask >= VL))
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;
            CTYPE temp = state[basis_index_0];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_1] = temp;
        }
    }  // if ((dim > VL) && (min_qubit_mask >= VL))
}
#endif

#ifdef _USE_MPI
void SWAP_gate_mpi(UINT target_qubit_index_0, UINT target_qubit_index_1,
    CTYPE* state, ITYPE dim, UINT inner_qc) {
    UINT left_qubit, right_qubit;
    if (target_qubit_index_0 > target_qubit_index_1) {
        left_qubit = target_qubit_index_0;
        right_qubit = target_qubit_index_1;
    } else {
        left_qubit = target_qubit_index_1;
        right_qubit = target_qubit_index_0;
    }

    if (left_qubit < inner_qc) {  // both qubits are inner
        SWAP_gate(target_qubit_index_0, target_qubit_index_1, state, dim);
    } else if (right_qubit < inner_qc) {  // one target is outer
        MPIutil& m = MPIutil::get_inst();
        const UINT rank = m.get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* t = m.get_workarea(&dim_work, &num_work);
        const ITYPE tgt_rank_bit = 1 << (left_qubit - inner_qc);
        const ITYPE rtgt_blk_dim = 1 << right_qubit;
        const int pair_rank = rank ^ tgt_rank_bit;

        ITYPE rtgt_offset = 0;
        if ((rank & tgt_rank_bit) == 0) rtgt_offset = rtgt_blk_dim;

        if (rtgt_blk_dim < dim_work) {
            dim_work >>= 1;  // upper for send, lower for recv
            CTYPE* t_send = t;
            CTYPE* t_recv = t + dim_work;
            const ITYPE num_rtgt_block = (dim / dim_work) >> 1;
            const ITYPE num_elem_block = dim_work >> right_qubit;

            CTYPE* si0 = state + rtgt_offset;
            for (ITYPE i = 0; i < num_rtgt_block; ++i) {
                // gather
                CTYPE* si = si0;
                CTYPE* ti = t_send;
                for (ITYPE k = 0; k < num_elem_block; ++k) {
                    memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));
                    si += (rtgt_blk_dim << 1);
                    ti += rtgt_blk_dim;
                }

                // sendrecv
                m.m_DC_sendrecv(t_send, t_recv, dim_work, pair_rank);

                // scatter
                si = t_recv;
                ti = si0;
                for (ITYPE k = 0; k < num_elem_block; ++k) {
                    memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));
                    si += rtgt_blk_dim;
                    ti += (rtgt_blk_dim << 1);
                }
                si0 += (dim_work << 1);
            }

        } else {  // rtgt_blk_dim >= dim_work
            const ITYPE num_rtgt_block = dim >> (right_qubit + 1);
            const ITYPE num_work_block = rtgt_blk_dim / dim_work;

            CTYPE* si = state + rtgt_offset;
            for (ITYPE i = 0; i < num_rtgt_block; ++i) {
                for (ITYPE j = 0; j < num_work_block; ++j) {
                    m.m_DC_sendrecv(si, t, dim_work, pair_rank);
                    memcpy(si, t, dim_work * sizeof(CTYPE));
                    si += dim_work;
                }
                si += rtgt_blk_dim;
            }
        }
    } else {  // both targets are outer
        MPIutil& m = MPIutil::get_inst();
        const UINT rank = m.get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* t = m.get_workarea(&dim_work, &num_work);
        const UINT tgt0_rank_bit = 1 << (left_qubit - inner_qc);
        const UINT tgt1_rank_bit = 1 << (right_qubit - inner_qc);
        const UINT tgt_rank_bit = tgt0_rank_bit + tgt1_rank_bit;

        const int pair_rank = rank ^ tgt_rank_bit;
        const int not_zerozero = ((rank & tgt_rank_bit) != 0);
        const int with_zero =
            (((rank & tgt0_rank_bit) * (rank & tgt1_rank_bit)) == 0);

        CTYPE* si = state;
        for (ITYPE i = 0; i < num_work; ++i) {
            if (not_zerozero && with_zero) {  // 01 or 10
                m.m_DC_sendrecv(si, t, dim_work, pair_rank);
                memcpy(si, t, dim_work * sizeof(CTYPE));
                si += dim_work;
            } else {
                m.get_tag();  // dummy to count up tag
            }
        }
    }
}
#endif
