
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void create_shift_mask_list_from_list_buf(
    const UINT* array, UINT count, UINT* dst_array, ITYPE* dst_mask);

void multi_qubit_dense_matrix_gate(const UINT* target_qubit_index_list,
    UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state,
    ITYPE dim) {
    if (target_qubit_index_count == 1) {
        single_qubit_dense_matrix_gate(
            target_qubit_index_list[0], matrix, state, dim);
    } else if (target_qubit_index_count == 2) {
        double_qubit_dense_matrix_gate_c(target_qubit_index_list[0],
            target_qubit_index_list[1], matrix, state, dim);
    } else {
#ifdef _OPENMP
        OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#endif
        multi_qubit_dense_matrix_gate_parallel(target_qubit_index_list,
            target_qubit_index_count, matrix, state, dim);
#ifdef _OPENMP
        OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    }
}

void create_shift_mask_list_from_list_buf(
    const UINT* array, UINT count, UINT* dst_array, ITYPE* dst_mask) {
    memcpy(dst_array, array, sizeof(UINT) * count);
    sort_ui(dst_array, count);
    for (UINT i = 0; i < count; ++i) {
        dst_mask[i] = (1UL << dst_array[i]) - 1;
    }
}

void multi_qubit_dense_matrix_gate_parallel(const UINT* target_qubit_index_list,
    UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state,
    ITYPE dim) {
    UINT sort_array[64];
    ITYPE mask_array[64];
    create_shift_mask_list_from_list_buf(target_qubit_index_list,
        target_qubit_index_count, sort_array, mask_array);

    // matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    const ITYPE* matrix_mask_list = create_matrix_mask_list(
        target_qubit_index_list, target_qubit_index_count);
    // loop variables
    const ITYPE loop_dim = dim >> target_qubit_index_count;

#ifdef _OPENMP
    const UINT thread_count = omp_get_max_threads();
#else
    const UINT thread_count = 1;
#endif
    CTYPE* buffer_list =
        (CTYPE*)malloc((size_t)(sizeof(CTYPE) * matrix_dim * thread_count));

    const ITYPE block_size = loop_dim / thread_count;
    const ITYPE residual = loop_dim % thread_count;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
        UINT thread_id = omp_get_thread_num();
#else
        UINT thread_id = 0;
#endif
        ITYPE start_index = block_size * thread_id +
                            (residual > thread_id ? thread_id : residual);
        ITYPE end_index =
            block_size * (thread_id + 1) +
            (residual > (thread_id + 1) ? (thread_id + 1) : residual);
        CTYPE* buffer = buffer_list + thread_id * matrix_dim;

        ITYPE state_index;
        for (state_index = start_index; state_index < end_index;
             ++state_index) {
            // create base index
            ITYPE basis_0 = state_index;
            for (UINT cursor = 0; cursor < target_qubit_index_count; ++cursor) {
                basis_0 = (basis_0 & mask_array[cursor]) +
                          ((basis_0 & (~mask_array[cursor])) << 1);
            }

            // compute matrix-vector multiply
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
    }
    free(buffer_list);
    free((ITYPE*)matrix_mask_list);
}
