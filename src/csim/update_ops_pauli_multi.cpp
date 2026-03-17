
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "constant.hpp"
#include "update_ops.hpp"
#include "utility.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _USE_MPI
#include "MPIutil.hpp"
#endif

/**
 * perform multi_qubit_Pauli_gate with XZ mask.
 *
 * This function assumes bit_flip_mask is not 0, i.e., at least one bit is
 * flipped. If no bit is flipped, use multi_qubit_Pauli_gate_Z_mask. This
 * function update the quantum state with Pauli operation. bit_flip_mask,
 * phase_flip_mask, global_phase_90rot_count, and pivot_qubit_index must be
 * computed before calling this function. See get_masks_from_*_list for the
 * above four arguemnts.
 */
void multi_qubit_Pauli_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask,
    UINT global_phase_90rot_count, UINT pivot_qubit_index, CTYPE* state,
    ITYPE dim);
void multi_qubit_Pauli_rotation_gate_XZ_mask(ITYPE bit_flip_mask,
    ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, double angle, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_gate_Z_mask(
    ITYPE phase_flip_mask, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_rotation_gate_Z_mask(
    ITYPE phase_flip_mask, double angle, CTYPE* state, ITYPE dim);

void multi_qubit_Pauli_gate_XZ_mask_single_thread(ITYPE bit_flip_mask,
    ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_rotation_gate_XZ_mask_single_thread(ITYPE bit_flip_mask,
    ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, double angle, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_gate_Z_mask_single_thread(
    ITYPE phase_flip_mask, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_rotation_gate_Z_mask_single_thread(
    ITYPE phase_flip_mask, double angle, CTYPE* state, ITYPE dim);

void multi_qubit_Pauli_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask,
    UINT global_phase_90rot_count, UINT pivot_qubit_index, CTYPE* state,
    ITYPE dim) {
    // loop varaibles
    const ITYPE loop_dim = dim / 2;
    ITYPE state_index;

    const ITYPE mask = (1ULL << pivot_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 14);
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        // create base index
        ITYPE basis_0 =
            (state_index & mask_low) + ((state_index & mask_high) << 1);

        // gather index
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;

        // determine sign
        UINT sign_0 = count_population(basis_0 & phase_flip_mask) % 2;
        UINT sign_1 = count_population(basis_1 & phase_flip_mask) % 2;

        // fetch values
        CTYPE cval_0 = state[basis_0];
        CTYPE cval_1 = state[basis_1];

        // set values
        state[basis_0] =
            cval_1 * PHASE_M90ROT[(global_phase_90rot_count + sign_0 * 2) % 4];
        state[basis_1] =
            cval_0 * PHASE_M90ROT[(global_phase_90rot_count + sign_1 * 2) % 4];
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}
void multi_qubit_Pauli_rotation_gate_XZ_mask(ITYPE bit_flip_mask,
    ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, double angle, CTYPE* state, ITYPE dim) {
    // loop varaibles
    const ITYPE loop_dim = dim / 2;
    ITYPE state_index;

    const ITYPE mask = (1ULL << pivot_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    // coefs
    const double cosval = cos(angle / 2);
    const double sinval = sin(angle / 2);
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 14);
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        // create base index
        ITYPE basis_0 =
            (state_index & mask_low) + ((state_index & mask_high) << 1);

        // gather index
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;

        // determine parity
        int bit_parity_0 = count_population(basis_0 & phase_flip_mask) % 2;
        int bit_parity_1 = count_population(basis_1 & phase_flip_mask) % 2;

        // fetch values
        CTYPE cval_0 = state[basis_0];
        CTYPE cval_1 = state[basis_1];

        // set values
        state[basis_0] =
            cosval * cval_0 +
            1.i * sinval * cval_1 *
                PHASE_M90ROT[(global_phase_90rot_count + bit_parity_0 * 2) % 4];
        state[basis_1] =
            cosval * cval_1 +
            1.i * sinval * cval_0 *
                PHASE_M90ROT[(global_phase_90rot_count + bit_parity_1 * 2) % 4];
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void multi_qubit_Pauli_gate_Z_mask(
    ITYPE phase_flip_mask, CTYPE* state, ITYPE dim) {
    // loop varaibles
    const ITYPE loop_dim = dim;
    ITYPE state_index;

#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 14);
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        // determine parity
        int bit_parity = count_population(state_index & phase_flip_mask) % 2;

        // set values
        if (bit_parity % 2 == 1) {
            state[state_index] *= -1;
        }
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void multi_qubit_Pauli_rotation_gate_Z_mask(
    ITYPE phase_flip_mask, double angle, CTYPE* state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;

    // coefs
    const double cosval = cos(angle / 2);
    const double sinval = sin(angle / 2);

#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 14);
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        // determine sign
        int bit_parity = count_population(state_index & phase_flip_mask) % 2;
        int sign = 1 - 2 * bit_parity;

        // set value
        state[state_index] *= cosval + (CTYPE)sign * 1.i * sinval;
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void multi_qubit_Pauli_gate_partial_list(const UINT* target_qubit_index_list,
    const UINT* Pauli_operator_type_list, UINT target_qubit_index_count,
    CTYPE* state, ITYPE dim) {
    // create pauli mask and call function
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    if (bit_flip_mask == 0) {
        multi_qubit_Pauli_gate_Z_mask(phase_flip_mask, state, dim);
    } else {
        multi_qubit_Pauli_gate_XZ_mask(bit_flip_mask, phase_flip_mask,
            global_phase_90rot_count, pivot_qubit_index, state, dim);
    }
}

void multi_qubit_Pauli_gate_whole_list(const UINT* Pauli_operator_type_list,
    UINT qubit_count, CTYPE* state, ITYPE dim) {
    // create pauli mask and call function
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count,
        &pivot_qubit_index);
    if (bit_flip_mask == 0) {
        multi_qubit_Pauli_gate_Z_mask(phase_flip_mask, state, dim);
    } else {
        multi_qubit_Pauli_gate_XZ_mask(bit_flip_mask, phase_flip_mask,
            global_phase_90rot_count, pivot_qubit_index, state, dim);
    }
}

void multi_qubit_Pauli_rotation_gate_partial_list(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, double angle, CTYPE* state, ITYPE dim) {
    // create pauli mask and call function
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    if (bit_flip_mask == 0) {
        multi_qubit_Pauli_rotation_gate_Z_mask(
            phase_flip_mask, angle, state, dim);
    } else {
        multi_qubit_Pauli_rotation_gate_XZ_mask(bit_flip_mask, phase_flip_mask,
            global_phase_90rot_count, pivot_qubit_index, angle, state, dim);
    }
}

void multi_qubit_Pauli_rotation_gate_whole_list(
    const UINT* Pauli_operator_type_list, UINT qubit_count, double angle,
    CTYPE* state, ITYPE dim) {
    // create pauli mask and call function
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count,
        &pivot_qubit_index);
    if (bit_flip_mask == 0) {
        multi_qubit_Pauli_rotation_gate_Z_mask(
            phase_flip_mask, angle, state, dim);
    } else {
        multi_qubit_Pauli_rotation_gate_XZ_mask(bit_flip_mask, phase_flip_mask,
            global_phase_90rot_count, pivot_qubit_index, angle, state, dim);
    }
}

void multi_qubit_Pauli_gate_XZ_mask_single_thread(ITYPE bit_flip_mask,
    ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, CTYPE* state, ITYPE dim) {
    // loop varaibles
    const ITYPE loop_dim = dim / 2;
    ITYPE state_index;

    const ITYPE mask = (1ULL << pivot_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    for (state_index = 0; state_index < loop_dim; ++state_index) {
        // create base index
        ITYPE basis_0 =
            (state_index & mask_low) + ((state_index & mask_high) << 1);

        // gather index
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;

        // determine sign
        UINT sign_0 = count_population(basis_0 & phase_flip_mask) % 2;
        UINT sign_1 = count_population(basis_1 & phase_flip_mask) % 2;

        // fetch values
        CTYPE cval_0 = state[basis_0];
        CTYPE cval_1 = state[basis_1];

        // set values
        state[basis_0] =
            cval_1 * PHASE_M90ROT[(global_phase_90rot_count + sign_0 * 2) % 4];
        state[basis_1] =
            cval_0 * PHASE_M90ROT[(global_phase_90rot_count + sign_1 * 2) % 4];
    }
}

void multi_qubit_Pauli_rotation_gate_XZ_mask_single_thread(ITYPE bit_flip_mask,
    ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, double angle, CTYPE* state, ITYPE dim) {
    // loop varaibles
    const ITYPE loop_dim = dim / 2;
    ITYPE state_index;

    const ITYPE mask = (1ULL << pivot_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    // coefs
    const double cosval = cos(angle / 2);
    const double sinval = sin(angle / 2);
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        // create base index
        ITYPE basis_0 =
            (state_index & mask_low) + ((state_index & mask_high) << 1);

        // gather index
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;

        // determine parity
        int bit_parity_0 = count_population(basis_0 & phase_flip_mask) % 2;
        int bit_parity_1 = count_population(basis_1 & phase_flip_mask) % 2;

        // fetch values
        CTYPE cval_0 = state[basis_0];
        CTYPE cval_1 = state[basis_1];

        // set values
        state[basis_0] =
            cosval * cval_0 +
            1.i * sinval * cval_1 *
                PHASE_M90ROT[(global_phase_90rot_count + bit_parity_0 * 2) % 4];
        state[basis_1] =
            cosval * cval_1 +
            1.i * sinval * cval_0 *
                PHASE_M90ROT[(global_phase_90rot_count + bit_parity_1 * 2) % 4];
    }
}

void multi_qubit_Pauli_gate_Z_mask_single_thread(
    ITYPE phase_flip_mask, CTYPE* state, ITYPE dim) {
    // loop varaibles
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        // determine parity
        int bit_parity = count_population(state_index & phase_flip_mask) % 2;

        // set values
        if (bit_parity % 2 == 1) {
            state[state_index] *= -1;
        }
    }
}

void multi_qubit_Pauli_rotation_gate_Z_mask_single_thread(
    ITYPE phase_flip_mask, double angle, CTYPE* state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;

    // coefs
    const double cosval = cos(angle / 2);
    const double sinval = sin(angle / 2);
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        // determine sign
        int bit_parity = count_population(state_index & phase_flip_mask) % 2;
        int sign = 1 - 2 * bit_parity;

        // set value
        state[state_index] *= cosval + (CTYPE)sign * 1.i * sinval;
    }
}

void multi_qubit_Pauli_gate_partial_list_single_thread(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, CTYPE* state, ITYPE dim) {
    // create pauli mask and call function
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    if (bit_flip_mask == 0) {
        multi_qubit_Pauli_gate_Z_mask_single_thread(
            phase_flip_mask, state, dim);
    } else {
        multi_qubit_Pauli_gate_XZ_mask_single_thread(bit_flip_mask,
            phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state,
            dim);
    }
}

void multi_qubit_Pauli_gate_whole_list_single_thread(
    const UINT* Pauli_operator_type_list, UINT qubit_count, CTYPE* state,
    ITYPE dim) {
    // create pauli mask and call function
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count,
        &pivot_qubit_index);
    if (bit_flip_mask == 0) {
        multi_qubit_Pauli_gate_Z_mask_single_thread(
            phase_flip_mask, state, dim);
    } else {
        multi_qubit_Pauli_gate_XZ_mask_single_thread(bit_flip_mask,
            phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state,
            dim);
    }
}

void multi_qubit_Pauli_rotation_gate_partial_list_single_thread(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, double angle, CTYPE* state, ITYPE dim) {
    // create pauli mask and call function
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    if (bit_flip_mask == 0) {
        multi_qubit_Pauli_rotation_gate_Z_mask_single_thread(
            phase_flip_mask, angle, state, dim);
    } else {
        multi_qubit_Pauli_rotation_gate_XZ_mask_single_thread(bit_flip_mask,
            phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle,
            state, dim);
    }
}

void multi_qubit_Pauli_rotation_gate_whole_list_single_thread(
    const UINT* Pauli_operator_type_list, UINT qubit_count, double angle,
    CTYPE* state, ITYPE dim) {
    // create pauli mask and call function
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count,
        &pivot_qubit_index);
    if (bit_flip_mask == 0) {
        multi_qubit_Pauli_rotation_gate_Z_mask_single_thread(
            phase_flip_mask, angle, state, dim);
    } else {
        multi_qubit_Pauli_rotation_gate_XZ_mask_single_thread(bit_flip_mask,
            phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle,
            state, dim);
    }
}

#ifdef _USE_MPI
/**
 * MPI-aware multi-qubit Pauli gate.
 *
 * Handles global qubits (those >= inner_qc, whose values are encoded in the
 * MPI rank number) by using sendrecv to exchange amplitudes with the partner
 * rank that holds the other half of each amplitude pair.
 *
 * Three cases:
 *   A) No global X/Y qubits: all pairs are local; delegate to the serial
 *      implementation with the global-Z rank-parity folded into the phase.
 *   B1) Global X/Y, no local X/Y: pair j with recv[j] from partner rank.
 *   B2) Mixed local+global X/Y: pair j with recv[j^local_bit_flip_mask].
 *       Requires local_bit_flip_mask < dim_work (pairs within one work chunk).
 */
void multi_qubit_Pauli_gate_partial_list_mpi(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, CTYPE* state, ITYPE dim, UINT inner_qc) {
    // Build masks (identical to the local wrapper)
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);

    // Partition at the MPI boundary
    const ITYPE inner_mask = (1ULL << inner_qc) - 1;
    const ITYPE local_bit_flip_mask = bit_flip_mask & inner_mask;
    const ITYPE global_bit_flip_rank_bits = bit_flip_mask >> inner_qc;
    const ITYPE local_phase_flip_mask = phase_flip_mask & inner_mask;
    const ITYPE global_phase_flip_rank_bits = phase_flip_mask >> inner_qc;

    const UINT rank = (UINT)MPIutil::get_inst().get_rank();
    const UINT global_rank_parity =
        (UINT)(count_population((ITYPE)rank & global_phase_flip_rank_bits) % 2);

    // ---- Case A: no global X/Y qubits: no MPI communication needed ----
    if (global_bit_flip_rank_bits == 0) {
        const UINT adj_count =
            (global_phase_90rot_count + 2 * global_rank_parity) % 4;
        if (bit_flip_mask == 0) {
            // Pure Z: global_phase_90rot_count == 0 here, so adj_count is 0
            // (global_rank_parity==0) or 2 (global_rank_parity==1).
            // Total sign for amplitude j: (-1)^(global_rank_parity + local_bp)
            if (adj_count % 4 == 0) {
                multi_qubit_Pauli_gate_Z_mask(
                    local_phase_flip_mask, state, dim);
            } else {
                // global_rank_parity == 1: negate where local parity is even
                // (total parity = 1+0 = odd -> negate)
#pragma omp parallel for
                for (ITYPE j = 0; j < dim; ++j) {
                    int bp =
                        (int)(count_population(j & local_phase_flip_mask) % 2);
                    if (bp == 0) state[j] *= -1;
                }
            }
        } else {
            multi_qubit_Pauli_gate_XZ_mask(local_bit_flip_mask,
                local_phase_flip_mask, adj_count, pivot_qubit_index, state,
                dim);
        }
        return;
    }

    // ---- Case B: global X/Y qubits - sendrecv with partner rank ----
    const UINT pair_rank = rank ^ (UINT)global_bit_flip_rank_bits;
    MPIutil& m = MPIutil::get_inst();
    ITYPE dim_work = dim;
    ITYPE num_work = 0;
    CTYPE* ptr_recv = m.get_workarea(&dim_work, &num_work);
    assert(num_work > 0);

    const UINT adj_count =
        (global_phase_90rot_count + 2 * global_rank_parity) % 4;

    // Decompose local_bit_flip_mask across the chunk boundary (_NQUBIT_WORK).
    // lbfm_high != 0 means a local X/Y qubit lives at bit >= _NQUBIT_WORK,
    // so its partner element is in a different chunk.
    const ITYPE lbfm_high = local_bit_flip_mask >> _NQUBIT_WORK;
    const ITYPE lbfm_low = local_bit_flip_mask & (dim_work - 1);

#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif
    if (lbfm_high == 0) {
        // ---- Case B same-chunk: partner element is within the same chunk ----
        CTYPE* ptr_state = state;
        for (ITYPE iter = 0; iter < num_work; ++iter) {
            m.m_DC_sendrecv(ptr_state, ptr_recv, dim_work, pair_rank);
            // Use the global local index for correct phase computation when
            // local_phase_flip_mask has bits >= _NQUBIT_WORK.
            const ITYPE jg_base = iter << _NQUBIT_WORK;
            if (local_bit_flip_mask == 0) {
                // B1: all X/Y global - pair j with recv[j]
#pragma omp parallel for
                for (ITYPE j = 0; j < dim_work; ++j) {
                    int bp = (int)(count_population(
                                       (jg_base | j) & local_phase_flip_mask) %
                                   2);
                    ptr_state[j] = ptr_recv[j] *
                                   PHASE_M90ROT[(adj_count + (UINT)bp * 2) % 4];
                }
            } else {
                // B2 (same-chunk): pair j with recv[j ^ lbfm_low]
                const UINT local_pivot =
                    (UINT)__builtin_ctzll((unsigned long long)lbfm_low);
                const ITYPE lp_mask = 1ULL << local_pivot;
                const ITYPE lp_mask_low = lp_mask - 1;
                const ITYPE lp_mask_high = ~lp_mask_low;
                const ITYPE loop_dim = dim_work / 2;
#pragma omp parallel for
                for (ITYPE si = 0; si < loop_dim; ++si) {
                    const ITYPE j =
                        (si & lp_mask_low) + ((si & lp_mask_high) << 1);
                    const ITYPE j_pair = j ^ lbfm_low;
                    int bp_j = (int)(count_population((jg_base | j) &
                                                      local_phase_flip_mask) %
                                     2);
                    int bp_jp = (int)(count_population((jg_base | j_pair) &
                                                       local_phase_flip_mask) %
                                      2);
                    ptr_state[j] =
                        ptr_recv[j_pair] *
                        PHASE_M90ROT[(adj_count + (UINT)bp_j * 2) % 4];
                    ptr_state[j_pair] =
                        ptr_recv[j] *
                        PHASE_M90ROT[(adj_count + (UINT)bp_jp * 2) % 4];
                }
            }
            ptr_state += dim_work;
        }
    } else {
        // ---- Case B cross-chunk: local X/Y qubit index >= _NQUBIT_WORK ----
        // Element at global-local index g = iter*dim_work + j pairs with
        //   g ^ local_bit_flip_mask = (iter^lbfm_high)*dim_work + (j^lbfm_low)
        // i.e., a different chunk. Process chunk pairs (iter, iter^lbfm_high)
        // together with two sendrecvs so both ranks issue matched calls:
        //   sendrecv(our[iter])     -> recv_a = partner[iter]
        //   sendrecv(our[iter_xor]) -> recv_b = partner[iter_xor]
        // then: our[iter][j]     <- recv_b[j^lbfm_low]
        //        our[iter_xor][j] <- recv_a[j^lbfm_low]
        //
        // lbfm_high != 0 is guaranteed here, so __builtin_clzll is safe.
        const ITYPE lbfm_high_msb =
            (ITYPE)1 << (sizeof(ITYPE) * 8 - 1 -
                         __builtin_clzll((unsigned long long)lbfm_high));
        std::vector<CTYPE> recv_b_vec(dim_work);
        CTYPE* const ptr_recv_b = recv_b_vec.data();
        for (ITYPE iter = 0; iter < num_work; ++iter) {
            if (iter & lbfm_high_msb) continue;  // skip second-of-pair
            const ITYPE iter_xor = iter ^ lbfm_high;
            CTYPE* const ptr_A = state + iter * dim_work;
            CTYPE* const ptr_B = state + iter_xor * dim_work;
            m.m_DC_sendrecv(ptr_A, ptr_recv, dim_work, pair_rank);
            m.m_DC_sendrecv(ptr_B, ptr_recv_b, dim_work, pair_rank);
            const ITYPE jg_base_A = iter << _NQUBIT_WORK;
            const ITYPE jg_base_B = iter_xor << _NQUBIT_WORK;
#pragma omp parallel for
            for (ITYPE j = 0; j < dim_work; ++j) {
                const ITYPE jp = j ^ lbfm_low;
                int bp_A = (int)(count_population(
                                     (jg_base_A | j) & local_phase_flip_mask) %
                                 2);
                int bp_B = (int)(count_population(
                                     (jg_base_B | j) & local_phase_flip_mask) %
                                 2);
                // our[iter][j]     <- partner[iter_xor][jp]  (in recv_b)
                ptr_A[j] = ptr_recv_b[jp] *
                           PHASE_M90ROT[(adj_count + (UINT)bp_A * 2) % 4];
                // our[iter_xor][j] <- partner[iter][jp]      (in recv_a)
                ptr_B[j] = ptr_recv[jp] *
                           PHASE_M90ROT[(adj_count + (UINT)bp_B * 2) % 4];
            }
        }
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

/**
 * MPI-aware multi-qubit Pauli rotation gate exp(-i*angle/2 * P).
 *
 * Same three-case structure as multi_qubit_Pauli_gate_partial_list_mpi but
 * applies the rotation formula instead of a plain swap.
 */
void multi_qubit_Pauli_rotation_gate_partial_list_mpi(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, double angle, CTYPE* state, ITYPE dim,
    UINT inner_qc) {
    // Build masks
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);

    // Partition at the MPI boundary
    const ITYPE inner_mask = (1ULL << inner_qc) - 1;
    const ITYPE local_bit_flip_mask = bit_flip_mask & inner_mask;
    const ITYPE global_bit_flip_rank_bits = bit_flip_mask >> inner_qc;
    const ITYPE local_phase_flip_mask = phase_flip_mask & inner_mask;
    const ITYPE global_phase_flip_rank_bits = phase_flip_mask >> inner_qc;

    const UINT rank = (UINT)MPIutil::get_inst().get_rank();
    const UINT global_rank_parity =
        (UINT)(count_population((ITYPE)rank & global_phase_flip_rank_bits) % 2);

    // ---- Case A: no global X/Y qubits - no MPI communication needed ----
    if (global_bit_flip_rank_bits == 0) {
        const UINT adj_count =
            (global_phase_90rot_count + 2 * global_rank_parity) % 4;
        if (bit_flip_mask == 0) {
            // Pure Z rotation: negate angle if global rank parity is odd
            double adj_angle = (global_rank_parity % 2) ? -angle : angle;
            multi_qubit_Pauli_rotation_gate_Z_mask(
                local_phase_flip_mask, adj_angle, state, dim);
        } else {
            // XZ rotation: pivot is guaranteed < inner_qc when no global X/Y
            multi_qubit_Pauli_rotation_gate_XZ_mask(local_bit_flip_mask,
                local_phase_flip_mask, adj_count, pivot_qubit_index, angle,
                state, dim);
        }
        return;
    }

    // ---- Case B: global X/Y qubits - sendrecv with partner rank ----
    const UINT pair_rank = rank ^ (UINT)global_bit_flip_rank_bits;
    MPIutil& m = MPIutil::get_inst();
    ITYPE dim_work = dim;
    ITYPE num_work = 0;
    CTYPE* ptr_recv = m.get_workarea(&dim_work, &num_work);
    assert(num_work > 0);

    const UINT adj_count =
        (global_phase_90rot_count + 2 * global_rank_parity) % 4;
    const double cosval = cos(angle / 2);
    const double sinval = sin(angle / 2);

    // Decompose local_bit_flip_mask across the chunk boundary (_NQUBIT_WORK).
    // lbfm_high != 0 means a local X/Y qubit lives at bit >= _NQUBIT_WORK,
    // so its partner element is in a different chunk.
    const ITYPE lbfm_high = local_bit_flip_mask >> _NQUBIT_WORK;
    const ITYPE lbfm_low = local_bit_flip_mask & (dim_work - 1);

#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif
    if (lbfm_high == 0) {
        // ---- Case B same-chunk: partner element is within the same chunk ----
        CTYPE* ptr_state = state;
        for (ITYPE iter = 0; iter < num_work; ++iter) {
            m.m_DC_sendrecv(ptr_state, ptr_recv, dim_work, pair_rank);
            // Use the global local index for correct phase computation when
            // local_phase_flip_mask has bits >= _NQUBIT_WORK.
            const ITYPE jg_base = iter << _NQUBIT_WORK;
            if (local_bit_flip_mask == 0) {
                // B1: all X/Y global - pair state[j] with recv[j]
#pragma omp parallel for
                for (ITYPE j = 0; j < dim_work; ++j) {
                    int bp = (int)(count_population(
                                       (jg_base | j) & local_phase_flip_mask) %
                                   2);
                    CTYPE phase = PHASE_M90ROT[(adj_count + (UINT)bp * 2) % 4];
                    ptr_state[j] = cosval * ptr_state[j] +
                                   1.i * sinval * ptr_recv[j] * phase;
                }
            } else {
                // B2 (same-chunk): pair state[j] with recv[j ^ lbfm_low]
                const UINT local_pivot =
                    (UINT)__builtin_ctzll((unsigned long long)lbfm_low);
                const ITYPE lp_mask = 1ULL << local_pivot;
                const ITYPE lp_mask_low = lp_mask - 1;
                const ITYPE lp_mask_high = ~lp_mask_low;
                const ITYPE loop_dim = dim_work / 2;
#pragma omp parallel for
                for (ITYPE si = 0; si < loop_dim; ++si) {
                    const ITYPE j =
                        (si & lp_mask_low) + ((si & lp_mask_high) << 1);
                    const ITYPE j_pair = j ^ lbfm_low;
                    // Save originals before overwriting.
                    const CTYPE a = ptr_state[j];
                    const CTYPE b = ptr_state[j_pair];
                    int bp_j = (int)(count_population((jg_base | j) &
                                                      local_phase_flip_mask) %
                                     2);
                    int bp_jp = (int)(count_population((jg_base | j_pair) &
                                                       local_phase_flip_mask) %
                                      2);
                    ptr_state[j] =
                        cosval * a +
                        1.i * sinval * ptr_recv[j_pair] *
                            PHASE_M90ROT[(adj_count + (UINT)bp_j * 2) % 4];
                    ptr_state[j_pair] =
                        cosval * b +
                        1.i * sinval * ptr_recv[j] *
                            PHASE_M90ROT[(adj_count + (UINT)bp_jp * 2) % 4];
                }
            }
            ptr_state += dim_work;
        }
    } else {
        // ---- Case B cross-chunk: local X/Y qubit index >= _NQUBIT_WORK ----
        // Element at global-local index g = iter*dim_work + j pairs with
        //   g ^ local_bit_flip_mask = (iter^lbfm_high)*dim_work + (j^lbfm_low)
        // i.e., a different chunk. Process chunk pairs (iter, iter^lbfm_high)
        // together with two sendrecvs so both ranks issue matched calls:
        //   sendrecv(our[iter])     -> recv_a = partner[iter]
        //   sendrecv(our[iter_xor]) -> recv_b = partner[iter_xor]
        // then: our[iter][j]     <- cosval*our[iter][j]     + sinval*recv_b[jp]
        //        our[iter_xor][j] <- cosval*our[iter_xor][j] +
        //        sinval*recv_a[jp]
        //
        // lbfm_high != 0 is guaranteed here, so __builtin_clzll is safe.
        const ITYPE lbfm_high_msb =
            (ITYPE)1 << (sizeof(ITYPE) * 8 - 1 -
                         __builtin_clzll((unsigned long long)lbfm_high));
        std::vector<CTYPE> recv_b_vec(dim_work);
        CTYPE* const ptr_recv_b = recv_b_vec.data();
        for (ITYPE iter = 0; iter < num_work; ++iter) {
            if (iter & lbfm_high_msb) continue;  // skip second-of-pair
            const ITYPE iter_xor = iter ^ lbfm_high;
            CTYPE* const ptr_A = state + iter * dim_work;
            CTYPE* const ptr_B = state + iter_xor * dim_work;
            m.m_DC_sendrecv(ptr_A, ptr_recv, dim_work, pair_rank);
            m.m_DC_sendrecv(ptr_B, ptr_recv_b, dim_work, pair_rank);
            const ITYPE jg_base_A = iter << _NQUBIT_WORK;
            const ITYPE jg_base_B = iter_xor << _NQUBIT_WORK;
#pragma omp parallel for
            for (ITYPE j = 0; j < dim_work; ++j) {
                const ITYPE jp = j ^ lbfm_low;
                int bp_A = (int)(count_population(
                                     (jg_base_A | j) & local_phase_flip_mask) %
                                 2);
                int bp_B = (int)(count_population(
                                     (jg_base_B | j) & local_phase_flip_mask) %
                                 2);
                // our[iter][j]     <- cosval*old + sinval*partner[iter_xor][jp]
                ptr_A[j] = cosval * ptr_A[j] +
                           1.i * sinval * ptr_recv_b[jp] *
                               PHASE_M90ROT[(adj_count + (UINT)bp_A * 2) % 4];
                // our[iter_xor][j] <- cosval*old + sinval*partner[iter][jp]
                ptr_B[j] = cosval * ptr_B[j] +
                           1.i * sinval * ptr_recv[jp] *
                               PHASE_M90ROT[(adj_count + (UINT)bp_B * 2) % 4];
            }
        }
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}
#endif  // _USE_MPI
