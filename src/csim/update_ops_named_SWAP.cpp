/////////////////////////////////////

#include <iostream>

/////////////////////////////////////

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
#elif defined(_USE_SVE)  // SVE (Scalable Vector Extension).
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
    std::cout << "low_mask = " << low_mask << std::endl;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    std::cout << "mid_mask = " << mid_mask << std::endl;
    const ITYPE high_mask = ~(max_qubit_mask - 1);
    std::cout << "high_mask = " << high_mask << std::endl;

    ITYPE state_index = 0;

    if (target_qubit_index_0 == 0 || target_qubit_index_1 == 0) {
        std::cout << "Entra no if" << std::endl;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;

            std::cout << "basis_index_0 = " << basis_index_0 << " basis_index_1 = " << basis_index_1 << std::endl;

            CTYPE temp = state[basis_index_0];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_1] = temp;
        }
    } else {
        // a,a+1 is swapped to a^m, a^m+1, respectively
        std::cout << "Entra no else" << std::endl;

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;

            std::cout << "basis_index_0 = " << basis_index_0 << " basis_index_1 = " << basis_index_1 << std::endl;

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

/* #ifdef _USE_MPI
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

    if (left_qubit < inner_qc) {  // both qubits are inner (internos).
        SWAP_gate(target_qubit_index_0, target_qubit_index_1, state, dim);
    } else if (right_qubit < inner_qc) {  // one target is outer (distribuido).
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

        } else {
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
    } else {
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
#endif */

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
        std::cout << "rank = " << rank << std::endl;
        ITYPE dim_work = dim;
        std::cout << "dim_work = " << dim_work << std::endl;
        ITYPE num_work = 0;
        CTYPE* t = m.get_workarea(&dim_work, &num_work);
        const ITYPE tgt_rank_bit = 1 << (left_qubit - inner_qc);
        std::cout << "tgt_rank_bit = " << tgt_rank_bit << std::endl;
        const ITYPE rtgt_blk_dim = 1 << right_qubit;
        std::cout << "rtgt_blk_dim = " << rtgt_blk_dim << std::endl;
        const int pair_rank = rank ^ tgt_rank_bit;
        std::cout << "pair_rank = " << pair_rank << std::endl;

        ITYPE rtgt_offset = 0;
        if ((rank & tgt_rank_bit) == 0) rtgt_offset = rtgt_blk_dim;
        std::cout << "rtgt_offset = " << rtgt_offset << std::endl;

        if (rtgt_blk_dim < dim_work) {
            dim_work >>= 1;  // upper for send, lower for recv
            CTYPE* t_send = t;
            CTYPE* t_recv = t + dim_work;
            const ITYPE num_rtgt_block = (dim / dim_work) >> 1;
            std::cout << "num_rtgt_block = " << num_rtgt_block << std::endl;

            const ITYPE num_elem_block = dim_work >> right_qubit;
            std::cout << "num_elem_block = " << num_elem_block << std::endl;

            CTYPE* si0 = state + rtgt_offset;

            ////////////////////////////////////////////////////

            /* for (ITYPE i = 0; i < num_rtgt_block; ++i) {
                std::cout << "[rank " << rank << "] OUTER i=" << i
                          << " si0_offset=" << (si0 - state)
                          << " dim_work=" << dim_work
                          << " num_elem_block=" << num_elem_block
                          << " rtgt_blk_dim=" << rtgt_blk_dim
                          << std::endl;

                // gather
                std::cout << "[rank " << rank << "]  GATHER start (i=" << i <<
            ")" << std::endl; CTYPE* si = si0; CTYPE* ti = t_send; for (ITYPE k
            = 0; k < num_elem_block; ++k) { std::ptrdiff_t src_off = si - state;
                    std::ptrdiff_t dst_off = ti - t_send;
                    std::cout << "[rank " << rank << "]   gather k=" << k
                              << "  src state[" << src_off << "] = ";
                    std::cout << *si << std::endl;

                    memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));

                    std::cout << "[rank " << rank << "]    wrote to t_send[" <<
            dst_off << "]"; std::cout << " = ";
                    // print rtgt_blk_dim elements (usually small)
                    for (ITYPE x = 0; x < rtgt_blk_dim; ++x) {
                        if (x) std::cout << ", ";
                        std::cout << t_send[dst_off + x];
                    }
                    std::cout << std::endl;

                    si += (rtgt_blk_dim << 1);
                    ti += rtgt_blk_dim;
                }

                // show t_send (limited)
                std::cout << "[rank " << rank << "]  t_send (first " <<
            std::min<ITYPE>(dim_work, (ITYPE)8) << " elems):"; for (ITYPE idx =
            0; idx < std::min<ITYPE>(dim_work, (ITYPE)8); ++idx) { std::cout <<
            " " << t_send[idx];
                }
                std::cout << std::endl;

                // sendrecv
                std::cout << "[rank " << rank << "]  sendrecv: sending " <<
            dim_work
                          << " elements to/from pair_rank=" << pair_rank <<
            std::endl; m.m_DC_sendrecv(t_send, t_recv, dim_work, pair_rank);

                // show t_recv after communication
                std::cout << "[rank " << rank << "]  t_recv (first " <<
            std::min<ITYPE>(dim_work, (ITYPE)8) << " elems):"; for (ITYPE idx =
            0; idx < std::min<ITYPE>(dim_work, (ITYPE)8); ++idx) { std::cout <<
            " " << t_recv[idx];
                }
                std::cout << std::endl;

                // scatter
                std::cout << "[rank " << rank << "]  SCATTER start (i=" << i <<
            ")" << std::endl; si = t_recv; ti = si0; for (ITYPE k = 0; k <
            num_elem_block; ++k) { std::ptrdiff_t src_off = si - t_recv;
                    std::ptrdiff_t dst_off = ti - state;
                    std::cout << "[rank " << rank << "]   scatter k=" << k
                              << " src t_recv[" << src_off << "] = " << *si
                              << " -> state[" << dst_off << "] (before = " <<
            state[dst_off] << ")"
                              << std::endl;

                    memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));

                    std::cout << "[rank " << rank << "]    wrote to state[" <<
            dst_off << "] = "; for (ITYPE x = 0; x < rtgt_blk_dim; ++x) { if (x)
            std::cout << ", "; std::cout << state[dst_off + x];
                    }
                    std::cout << std::endl;

                    si += rtgt_blk_dim;
                    ti += (rtgt_blk_dim << 1);
                }
                si0 += (dim_work << 1);
                std::cout << "[rank " << rank << "]  end OUTER i=" << i
                          << " next si0_offset=" << (si0 - state) << std::endl;
            } */

            //////////////////////////////////////////////////////

            for (ITYPE i = 0; i < num_rtgt_block; ++i) {
                // gather
                CTYPE* si = si0;
                CTYPE* ti = t_send;
                for (ITYPE k = 0; k < num_elem_block; ++k) {
                    memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));
                    si += (rtgt_blk_dim << 1);
                    ti += rtgt_blk_dim;
                }

                std::cout << "[Rank " << rank
                          << "] Enviando a pair_rank=" << pair_rank
                          << " (dim_work=" << dim_work
                          << ") índices globales enviados: ";

                CTYPE* si_tmp = si0;
                for (ITYPE k = 0; k < std::min<ITYPE>(num_elem_block, (ITYPE)4);
                    ++k) {  // limitamos para no imprimir todo
                    std::ptrdiff_t offset = si_tmp - state;
                    std::cout << "[";
                    for (ITYPE x = 0;
                        x < std::min<ITYPE>(rtgt_blk_dim, (ITYPE)4);
                        ++x) {  // también limitamos cada bloque
                        std::cout << (offset + x);
                        if (x + 1 < std::min<ITYPE>(rtgt_blk_dim, (ITYPE)4))
                            std::cout << ",";
                    }
                    std::cout << "] ";
                    si_tmp += (rtgt_blk_dim << 1);
                }
                std::cout << std::endl;

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
        std::cout << "dim_work = " << dim_work << std::endl;
        ITYPE num_work = 0;
        CTYPE* t = m.get_workarea(&dim_work, &num_work);
        const UINT tgt0_rank_bit = 1 << (left_qubit - inner_qc);
        std::cout << "tgt0_rank_bit = " << tgt0_rank_bit << std::endl;
        const UINT tgt1_rank_bit = 1 << (right_qubit - inner_qc);
        std::cout << "tgt1_rank_bit = " << tgt1_rank_bit << std::endl;
        const UINT tgt_rank_bit = tgt0_rank_bit + tgt1_rank_bit;
        std::cout << "tgt_rank_bit = " << tgt_rank_bit << std::endl;

        std::cout << "rank = " << rank << std::endl;
        const int pair_rank = rank ^ tgt_rank_bit;
        std::cout << "pair_rank = " << pair_rank << std::endl;
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
