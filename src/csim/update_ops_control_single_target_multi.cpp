
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "MPIutil.hpp"
#include "constant.hpp"
#include "update_ops.hpp"
#include "utility.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void single_qubit_control_multi_qubit_dense_matrix_gate(
    UINT control_qubit_index, UINT control_value,
    const UINT* target_qubit_index_list, UINT target_qubit_index_count,
    const CTYPE* matrix, CTYPE* state, ITYPE dim) {
    // matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    ITYPE* matrix_mask_list = create_matrix_mask_list(
        target_qubit_index_list, target_qubit_index_count);
    CTYPE* buffer = (CTYPE*)malloc((size_t)(sizeof(CTYPE) * matrix_dim));

    // insert list
    const UINT insert_index_count = target_qubit_index_count + 1;
    UINT* sorted_insert_index_list = create_sorted_ui_list_value(
        target_qubit_index_list, target_qubit_index_count, control_qubit_index);

    // control mask
    const ITYPE control_mask = (1ULL << control_qubit_index) * control_value;

    // loop varaibles
    const ITYPE loop_dim = dim >> insert_index_count;
    ITYPE state_index;

    for (state_index = 0; state_index < loop_dim; ++state_index) {
        // create base index
        ITYPE basis_0 = state_index;
        for (UINT cursor = 0; cursor < insert_index_count; cursor++) {
            UINT insert_index = sorted_insert_index_list[cursor];
            basis_0 = insert_zero_to_basis_index(
                basis_0, 1ULL << insert_index, insert_index);
        }

        // flip control
        basis_0 ^= control_mask;

        // compute matrix mul
        for (ITYPE y = 0; y < matrix_dim; ++y) {
            buffer[y] = 0;
            for (ITYPE x = 0; x < matrix_dim; ++x) {
                buffer[y] += matrix[y * matrix_dim + x] *
                             state[basis_0 ^ matrix_mask_list[x]];
            }
        }

        // set result
        for (ITYPE y = 0; y < matrix_dim; ++y) {
            state[basis_0 ^ matrix_mask_list[y]] = buffer[y];
        }
    }
    free(sorted_insert_index_list);
    free(buffer);
    free(matrix_mask_list);
}

#ifdef _USE_MPI
void single_qubit_control_multi_qubit_dense_matrix_gate_mpi(
    UINT control_qubit_index, UINT control_value,
    const UINT* target_qubit_index_list, UINT target_qubit_index_count,
    const CTYPE* matrix, CTYPE* state, ITYPE dim, UINT inner_qc) {
    UINT* new_target_index_list =
        (UINT*)malloc((size_t)(sizeof(UINT) * target_qubit_index_count));

    // count outer-target qubit
    UINT num_outer_target = 0;
    for (UINT i = 0; i < target_qubit_index_count; ++i) {
        new_target_index_list[i] = target_qubit_index_list[i];
        if (target_qubit_index_list[i] >= inner_qc) ++num_outer_target;
    }

    // swap all outer qubits
    std::vector<UINT> inner_avail_qubit(inner_qc);
    std::vector<UINT> outer_target_index;
    UINT new_control_index = control_qubit_index;
    if (target_qubit_index_count > 0) {
        if (target_qubit_index_count > inner_qc) {
            throw NotImplementedException(
                "Dense Matrix single-control-multi-target gate for MPI with " +
                std::to_string(target_qubit_index_count) +
                " target-qubits and " + std::to_string(inner_qc) +
                " local-qubits is not Implemented");
        }

        std::vector<UINT> swapped_i;

        // set act_target_index
        std::iota(inner_avail_qubit.begin(), inner_avail_qubit.end(), 0);
        std::reverse(inner_avail_qubit.begin(), inner_avail_qubit.end());
        if (control_qubit_index < inner_qc) {
            std::remove(inner_avail_qubit.begin(), inner_avail_qubit.end(),
                control_qubit_index);
            inner_avail_qubit.data()[inner_qc - 1] = control_qubit_index;
        }
        for (UINT i = 0; i < target_qubit_index_count; ++i) {
            if (target_qubit_index_list[i] >= inner_qc) {
                outer_target_index.push_back(target_qubit_index_list[i]);
                swapped_i.push_back(i);
            } else {
                std::remove(inner_avail_qubit.begin(), inner_avail_qubit.end(),
                    target_qubit_index_list[i]);
            }
        }

        for (UINT i = 0; i < num_outer_target; ++i) {
            if (inner_avail_qubit[i] == control_qubit_index)
                new_control_index = outer_target_index[i];
            new_target_index_list[swapped_i[i]] = inner_avail_qubit[i];
        }
        // swap all outer qubits
        for (UINT i = 0; i < num_outer_target; ++i) {
            SWAP_gate_mpi(outer_target_index[i], inner_avail_qubit[i], state,
                dim, inner_qc);
        }
    }

    // update state-vector
    if (new_control_index < inner_qc) {
        single_qubit_control_multi_qubit_dense_matrix_gate(new_control_index,
            control_value, new_target_index_list, target_qubit_index_count,
            matrix, state, dim);
    } else {
        const UINT rank = MPIutil::get_inst().get_rank();
        const UINT control_rank_bit = 1 << (new_control_index - inner_qc);
        if ((rank & control_rank_bit) >> (new_control_index - inner_qc) ==
            control_value)
            multi_qubit_dense_matrix_gate_mpi(new_target_index_list,
                target_qubit_index_count, matrix, state, dim, inner_qc);
        // else, nothing to do
    }

    // revert all outer qubits
    if (num_outer_target > 0) {
        for (UINT i = 0; i < num_outer_target; ++i) {
            SWAP_gate_mpi(outer_target_index[i], inner_avail_qubit[i], state,
                dim, inner_qc);
        }
    }
    free(new_target_index_list);
}
#endif
