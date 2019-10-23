
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "constant.h"
#include "update_ops.h"
#include "utility.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void single_qubit_control_multi_qubit_dense_matrix_gate(UINT control_qubit_index, UINT control_value, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim) {

	// matrix dim, mask, buffer
	const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
	ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
	CTYPE* buffer = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*matrix_dim));

	// insert list
	const UINT insert_index_count = target_qubit_index_count + 1;
	UINT* sorted_insert_index_list = create_sorted_ui_list_value(target_qubit_index_list, target_qubit_index_count, control_qubit_index);

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
			basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index, insert_index);
		}

		// flip control
		basis_0 ^= control_mask;

		// compute matrix mul
		for (ITYPE y = 0; y < matrix_dim; ++y) {
			buffer[y] = 0;
			for (ITYPE x = 0; x < matrix_dim; ++x) {
				buffer[y] += matrix[y*matrix_dim + x] * state[basis_0 ^ matrix_mask_list[x]];
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
