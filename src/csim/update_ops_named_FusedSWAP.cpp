
#include <cassert>
#include <cstring>

#include "MPIutil.hpp"
#include "constant.hpp"
#include "memory_ops.hpp"
#include "update_ops.hpp"
#include "utility.hpp"

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void FusedSWAP_gate(UINT target_qubit_index_0, UINT target_qubit_index_1,
    UINT block_size, CTYPE* state, ITYPE dim) {
#if 0
    for (UINT i = 0; i < block_size; ++i) {
        SWAP_gate(
            target_qubit_index_0 + i, target_qubit_index_1 + i, state, dim);
    }
#else
    const UINT nqubits = count_population(dim - 1);
    UINT upper_index, lower_index;
    if (target_qubit_index_0 > target_qubit_index_1) {
        upper_index = target_qubit_index_0;
        lower_index = target_qubit_index_1;
    } else {
        upper_index = target_qubit_index_1;
        lower_index = target_qubit_index_0;
    }
    assert(upper_index > (lower_index + block_size - 1));
    assert(nqubits > (upper_index + block_size - 1));

    const ITYPE kblk_dim = 1ULL << (nqubits - upper_index);
    const ITYPE jblk_dim = 1ULL << (upper_index - lower_index);
    const ITYPE iblk_dim = 1ULL << lower_index;
    const ITYPE mask_block = (1 << block_size) - 1;

    for (ITYPE kblk = 0; kblk < kblk_dim; ++kblk) {
        const ITYPE kblk_masked = kblk & mask_block;
        const ITYPE kblk_head = kblk - kblk_masked;
        const ITYPE jblk_start = kblk_masked + 1;

        for (ITYPE jblk = jblk_start; jblk < jblk_dim; ++jblk) {
            const ITYPE jblk_masked = jblk & mask_block;
            const ITYPE jblk_head = jblk - jblk_masked;
            if (jblk_masked < jblk_start) continue;

            CTYPE* si = state + (kblk << upper_index) + (jblk << lower_index);
            CTYPE* ti = state + ((kblk_head + jblk_masked) << upper_index) +
                        ((jblk_head + kblk_masked) << lower_index);
            for (ITYPE i = 0; i < iblk_dim; ++i) {
                CTYPE tmp = *si;
                *si++ = *ti;
                *ti++ = tmp;
            }
        }
    }
#endif
}

#ifdef _USE_MPI
void FusedSWAP_gate_global(UINT target_qubit_index_0, UINT target_qubit_index_1,
    UINT block_size, CTYPE* state, ITYPE dim) {
    const UINT nqubits = count_population(dim - 1);
#if 0
    for (UINT i = 0; i < block_size; ++i) {
        SWAP_gate_mpi(
            target_qubit_index_0 + i, target_qubit_index_1 + i, state, dim, nqubits);
    }
#else
    assert(target_qubit_index_0 > nqubits);
    assert(target_qubit_index_1 > nqubits);
    assert(std::abs(static_cast<std::int32_t>(target_qubit_index_0) -
                    static_cast<std::int32_t>(target_qubit_index_1)) >=
           block_size);

    MPIutil& m = MPIutil::get_inst();
    const int rank = m.get_rank();

    const int blk0_idx = target_qubit_index_0 - nqubits;
    const int blk1_idx = target_qubit_index_1 - nqubits;
    const int mask_block = (1 << block_size) - 1;
    const int blk0_mask = mask_block << blk0_idx;
    const int blk1_mask = mask_block << blk1_idx;
    const int blk_mask = blk0_mask + blk1_mask;
    const int pair_rank0 = rank & (~blk_mask);

    const int pair_rank = pair_rank0 +
                          ((rank & blk0_mask) >> blk0_idx << blk1_idx) +
                          ((rank & blk1_mask) >> blk1_idx << blk0_idx);
    bool flag_exchange = (rank != pair_rank);

    ITYPE dim_work = dim;
    ITYPE num_work = 0;
    CTYPE* t = m.get_workarea(&dim_work, &num_work);
    assert(num_work > 0);

    CTYPE* si = state;
    for (ITYPE i = 0; i < num_work; ++i) {
        if (flag_exchange) {
            m.m_DC_sendrecv(si, t, dim_work, pair_rank);
            memcpy(si, t, dim_work * sizeof(CTYPE));
            si += dim_work;
        } else {
            m.get_tag();  // dummy to count up tag
        }
    }
#endif
}

static inline void _gather(CTYPE* t_send, const CTYPE* state, const UINT i,
    const ITYPE num_elem_block, const UINT rtgt_offset_index,
    const ITYPE rtgt_blk_dim, const UINT act_bs) {
    for (UINT k = 0; k < num_elem_block; ++k) {
        UINT iter = i * num_elem_block + k;
        const CTYPE* si =
            state + (rtgt_offset_index ^ (iter << act_bs)) * rtgt_blk_dim;
        CTYPE* ti = t_send + k * rtgt_blk_dim;
        memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));
    }
}

static inline void _scatter(CTYPE* state, const CTYPE* t_recv, const UINT i,
    const ITYPE num_elem_block, const UINT rtgt_offset_index,
    const ITYPE rtgt_blk_dim, const UINT act_bs) {
    for (UINT k = 0; k < num_elem_block; ++k) {
        UINT iter = i * num_elem_block + k;
        CTYPE* ti =
            state + (rtgt_offset_index ^ (iter << act_bs)) * rtgt_blk_dim;
        const CTYPE* si = t_recv + k * rtgt_blk_dim;
        memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));
    }
}

void FusedSWAP_gate_mpi(UINT target_qubit_index_0, UINT target_qubit_index_1,
    UINT block_size, CTYPE* state, ITYPE dim, UINT inner_qc) {
    // ex.
    // +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    // |19|18|17|16|15|14|13|12|11|10| 9| 8| 7| 6| 5| 4| 3| 2| 1| 0|
    // +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    // |<-- global qubits-->|<-------     local qubits     ------->|
    // +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    //                    ^^ target=0                ^^target1
    // Fused-SWAP (target0=13, target1=4, block size=4) is same as
    //     SWAP(13,4) + SWAP(14,5) + SWAP(15,6) + SWAP(16,7).
    //
    if (block_size == 0) return;
    UINT upper_index, lower_index;
    if (target_qubit_index_0 > target_qubit_index_1) {
        upper_index = target_qubit_index_0;
        lower_index = target_qubit_index_1;
    } else {
        upper_index = target_qubit_index_1;
        lower_index = target_qubit_index_0;
    }
    assert(upper_index > (lower_index + block_size - 1));

    UINT act_bs = block_size;

    if ((lower_index + block_size - 1) >= inner_qc) {
        const UINT num_outer_swap =
            std::min(lower_index + block_size - inner_qc, block_size);
        /* swap pairs of global qubits */
        FusedSWAP_gate_global(
            target_qubit_index_0 + block_size - num_outer_swap,
            target_qubit_index_1 + block_size - num_outer_swap, num_outer_swap,
            state, dim);
        act_bs -= num_outer_swap;
    }

    if (inner_qc > upper_index) {
        const UINT num_inner_swap =
            std::min(inner_qc - upper_index, block_size);
        /* swap pairs of local qubits */
        FusedSWAP_gate(target_qubit_index_0, target_qubit_index_1,
            num_inner_swap, state, dim);

        act_bs -= num_inner_swap;
        upper_index += num_inner_swap;
        lower_index += num_inner_swap;
    }

    if (act_bs == 0) {
        return;
    }

    /*  FusedSWAP main */
    /* All remained swaps are pairs of inner and outer */

    MPIutil& m = MPIutil::get_inst();
    const UINT rank = m.get_rank();
    ITYPE dim_work = dim;
    ITYPE num_work = 0;
    CTYPE* t = m.get_workarea(&dim_work, &num_work);
    const ITYPE rtgt_blk_dim = 1 << lower_index;
    const UINT num_pair = 1 << act_bs;
    const UINT tgt_outer_rank_gap = upper_index - inner_qc;
    const UINT tgt_inner_rank_gap = inner_qc - lower_index;

    // use two-workarea for double buffering
    dim_work = get_min_ll(dim_work, dim >> (act_bs - 1)) >> 1;

    if (rtgt_blk_dim < dim_work) {  // unit elems block smaller than worksize
        dim_work >>= 1;             // 1/2: for send, 1/2: for recv

        const ITYPE num_rtgt_block = (dim / dim_work) >> act_bs;
        const ITYPE num_elem_block = dim_work / rtgt_blk_dim;
        const UINT offset_mask = (1 << tgt_inner_rank_gap) - 1;

        CTYPE* send_buf[2] = {t + dim_work * 0, t + dim_work * 1};
        CTYPE* recv_buf[2] = {t + dim_work * 2, t + dim_work * 3};

        const ITYPE stepi_total = (num_pair - 1) * num_rtgt_block;
        ITYPE stepi = 0;
        UINT buf_idx = 0;

        if (0 < stepi_total) {
            const UINT step = 1;
            const UINT i = 0;
            const UINT pair_rank = rank ^ (step << tgt_outer_rank_gap);
            UINT rtgt_offset_index =
                ((rank >> tgt_outer_rank_gap) ^ step) & offset_mask;
            _gather(send_buf[buf_idx], state, i, num_elem_block,
                rtgt_offset_index, rtgt_blk_dim, act_bs);
            m.m_DC_isendrecv(
                send_buf[buf_idx], recv_buf[buf_idx], dim_work, pair_rank);
        }
        for (UINT step = 1; step < num_pair; ++step) {
            for (UINT i = 0; i < num_rtgt_block; ++i) {
                if ((stepi + 1) < stepi_total) {
                    UINT i_next = i + 1;
                    UINT s_next = step;
                    if (i_next >= num_rtgt_block) {
                        i_next = 0;
                        s_next++;
                    }
                    const UINT pair_rank =
                        rank ^ (s_next << tgt_outer_rank_gap);
                    const UINT rtgt_offset_index =
                        ((rank >> tgt_outer_rank_gap) ^ s_next) & offset_mask;

                    _gather(send_buf[buf_idx ^ 1], state, i_next,
                        num_elem_block, rtgt_offset_index, rtgt_blk_dim,
                        act_bs);
                    m.m_DC_isendrecv(send_buf[buf_idx ^ 1],
                        recv_buf[buf_idx ^ 1], dim_work, pair_rank);
                }

                const UINT rtgt_offset_index =
                    ((rank >> tgt_outer_rank_gap) ^ step) & offset_mask;
                m.mpi_wait(2);
                _scatter(state, recv_buf[buf_idx], i, num_elem_block,
                    rtgt_offset_index, rtgt_blk_dim, act_bs);
                buf_idx ^= 1;
                stepi++;
            }
        }
    } else {  // rtgt_blk_dim >= dim_work
        UINT TotalSizePerPairComm = dim >> act_bs;
        const ITYPE num_elem_block = TotalSizePerPairComm >> lower_index;
        assert((rtgt_blk_dim % dim_work) == 0);
        const ITYPE num_loop_per_block = rtgt_blk_dim / dim_work;
        UINT offset_mask = (1 << tgt_inner_rank_gap) - 1;

        CTYPE* buf[2] = {t, t + dim_work};

        const ITYPE sjk_total =
            (num_pair - 1) * num_elem_block * num_loop_per_block;
        ITYPE sjk = 0;
        UINT buf_idx = 0;
        if (0 < sjk_total) {  // first sendrecv
            const UINT step = 1;
            const ITYPE j = 0;
            const ITYPE k = 0;
            const UINT pair_rank = rank ^ (step << tgt_outer_rank_gap);
            const UINT rtgt_offset_index =
                ((rank >> tgt_outer_rank_gap) ^ step) & offset_mask;
            CTYPE* si =
                state + (rtgt_offset_index ^ (j << act_bs)) * rtgt_blk_dim;
            m.m_DC_isendrecv(
                si + dim_work * k, buf[buf_idx], dim_work, pair_rank);
        }
        for (UINT step = 1; step < num_pair; ++step) {  // pair communication
            for (ITYPE j = 0; j < num_elem_block; ++j) {
                for (ITYPE k = 0; k < num_loop_per_block; ++k) {
                    if ((sjk + 1) < sjk_total) {
                        ITYPE k_next = k + 1;
                        ITYPE j_next = j;
                        ITYPE s_next = step;
                        if (k_next >= num_loop_per_block) {
                            k_next = 0;
                            j_next++;
                        }
                        if (j_next >= num_elem_block) {
                            j_next = 0;
                            s_next++;
                        }
                        const UINT pair_rank =
                            rank ^ (s_next << tgt_outer_rank_gap);
                        const UINT rtgt_offset_index =
                            ((rank >> tgt_outer_rank_gap) ^ s_next) &
                            offset_mask;
                        CTYPE* si_next =
                            state +
                            (rtgt_offset_index ^ (j_next << act_bs)) *
                                rtgt_blk_dim +
                            dim_work * k_next;
                        m.m_DC_isendrecv(
                            si_next, buf[buf_idx ^ 1], dim_work, pair_rank);
                    }

                    const UINT rtgt_offset_index =
                        ((rank >> tgt_outer_rank_gap) ^ step) & offset_mask;
                    CTYPE* si =
                        state +
                        (rtgt_offset_index ^ (j << act_bs)) * rtgt_blk_dim +
                        dim_work * k;
                    m.mpi_wait(2);  // wait 2 async comm
                    memcpy(si, buf[buf_idx], dim_work * sizeof(CTYPE));
                    buf_idx ^= 1;
                    sjk++;
                }
            }
        }
    }
}
#endif
