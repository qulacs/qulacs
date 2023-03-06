
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

void multi_qubit_control_multi_qubit_dense_matrix_gate(
    const UINT* control_qubit_index_list, const UINT* control_value_list,
    UINT control_qubit_index_count, const UINT* target_qubit_index_list,
    UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state,
    ITYPE dim) {
    // matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    ITYPE* matrix_mask_list = create_matrix_mask_list(
        target_qubit_index_list, target_qubit_index_count);
    CTYPE* buffer = (CTYPE*)malloc((size_t)(sizeof(CTYPE) * matrix_dim));

    // insert index
    const UINT insert_index_count =
        target_qubit_index_count + control_qubit_index_count;
    UINT* sorted_insert_index_list = create_sorted_ui_list_list(
        target_qubit_index_list, target_qubit_index_count,
        control_qubit_index_list, control_qubit_index_count);

    // control mask
    ITYPE control_mask = create_control_mask(control_qubit_index_list,
        control_value_list, control_qubit_index_count);

    // loop varaibles
    const ITYPE loop_dim =
        dim >> (target_qubit_index_count + control_qubit_index_count);
    ITYPE state_index;

    for (state_index = 0; state_index < loop_dim; ++state_index) {
        // create base index
        ITYPE basis_0 = state_index;
        for (UINT cursor = 0; cursor < insert_index_count; cursor++) {
            UINT insert_index = sorted_insert_index_list[cursor];
            basis_0 = insert_zero_to_basis_index(
                basis_0, 1ULL << insert_index, insert_index);
        }

        // flip control masks
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
void multi_qubit_control_multi_qubit_dense_matrix_gate_mpi(
    const UINT* control_qubit_index_list, const UINT* control_value_list,
    UINT control_count, const UINT* target_qubit_index_list, UINT target_count,
    const CTYPE* matrix, CTYPE* state, ITYPE dim, UINT inner_qc) {
    UINT* index_list_buf = (UINT*)malloc(
        (size_t)(sizeof(UINT) * (control_count * 2 + target_count)));

    UINT* new_control_index_list = index_list_buf;
    UINT* new_control_value_list = index_list_buf + control_count;
    UINT* new_target_index_list = index_list_buf + control_count * 2;

    // copy control qubit lists
    for (UINT j = 0; j < control_count; ++j) {
        new_control_index_list[j] = control_qubit_index_list[j];
        new_control_value_list[j] = control_value_list[j];
    }
    // count outer-target qubits
    UINT num_outer_target = 0;
    for (UINT i = 0; i < target_count; ++i) {
        new_target_index_list[i] = target_qubit_index_list[i];
        if (target_qubit_index_list[i] >= inner_qc) ++num_outer_target;
    }

    // swap all outer target qubits to inner
    std::vector<UINT> inner_avail_qubit(inner_qc);
    std::vector<UINT> outer_target_index;
    if (num_outer_target > 0) {
        if (target_count > inner_qc) {
            throw NotImplementedException(
                "Dense Matrix multi-control-multi-target gate for MPI with " +
                std::to_string(target_count) + " target-qubits and " +
                std::to_string(inner_qc) + " local-qubits is not Implemented");
        }

        std::vector<UINT> swapped_i;

        // set act_target_index
        std::iota(inner_avail_qubit.begin(), inner_avail_qubit.end(), 0);
        std::reverse(inner_avail_qubit.begin(), inner_avail_qubit.end());
        for (UINT i = 0; i < control_count; ++i) {
            if (control_qubit_index_list[i] < inner_qc) {
                std::remove(inner_avail_qubit.begin(), inner_avail_qubit.end(),
                    control_qubit_index_list[i]);
                inner_avail_qubit.data()[inner_qc - 1] =
                    control_qubit_index_list[i];
            }
        }
        for (UINT i = 0; i < target_count; ++i) {
            if (target_qubit_index_list[i] >= inner_qc) {
                outer_target_index.push_back(target_qubit_index_list[i]);
                swapped_i.push_back(i);
            } else {
                std::remove(inner_avail_qubit.begin(), inner_avail_qubit.end(),
                    target_qubit_index_list[i]);
            }
        }

        for (UINT i = 0; i < num_outer_target; ++i) {
            for (UINT j = 0; j < control_count; ++j) {
                if (inner_avail_qubit[i] == control_qubit_index_list[j]) {
                    new_control_index_list[j] = new_control_index_list[i];
                    new_control_value_list[j] = new_control_value_list[i];
                }
                new_target_index_list[swapped_i[i]] = inner_avail_qubit[i];
            }
        }
        // swap all outer qubits
        for (UINT i = 0; i < num_outer_target; ++i) {
            SWAP_gate_mpi(outer_target_index[i], inner_avail_qubit[i], state,
                dim, inner_qc);
        }
    }

    // make global control flags
    UINT new_control_count = 0;
    UINT mask_control_global_0 = 0;
    UINT mask_control_global_1 = 0;
    for (UINT i = 0; i < control_count; ++i) {
        if (control_qubit_index_list[i] < inner_qc) {
            new_control_index_list[new_control_count] =
                new_control_index_list[i];
            new_control_value_list[new_control_count] =
                new_control_value_list[i];
            new_control_count++;
        } else {
            if (control_value_list[i] == 0)
                mask_control_global_0 |=
                    1 << (new_control_index_list[i] - inner_qc);
            else
                mask_control_global_1 |=
                    1 << (new_control_index_list[i] - inner_qc);
        }
    }

    // count outer-control qubit
    UINT num_outer_control = 0;
    for (UINT i = 0; i < control_count; ++i)
        if (new_control_index_list[i] >= inner_qc) ++num_outer_control;

    if ((mask_control_global_0 + mask_control_global_1) > 0) {
        // update state (some control qubits are in global)
        const UINT rank = MPIutil::get_inst().get_rank();
        if ((rank & mask_control_global_0) |
            ((~rank) & mask_control_global_1)) {  // do nothing
        } else {
            // update state (all control qubits are in local)
            multi_qubit_control_multi_qubit_dense_matrix_gate(
                new_control_index_list, new_control_value_list,
                new_control_count, new_target_index_list, target_count, matrix,
                state, dim);
        }
    } else {
        // update state (all control qubits are in local)
        multi_qubit_control_multi_qubit_dense_matrix_gate(
            new_control_index_list, new_control_value_list, new_control_count,
            new_target_index_list, target_count, matrix, state, dim);
    }

    // revert all outer qubits
    if (num_outer_target > 0) {
        // swap all outer qubits
        for (UINT i = 0; i < num_outer_target; ++i) {
            SWAP_gate_mpi(outer_target_index[i], inner_avail_qubit[i], state,
                dim, inner_qc);
        }
    }
    free(index_list_buf);
}
#endif
