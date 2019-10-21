
#include "update_ops_cpp.hpp"
#ifndef _MSC_VER
extern "C" {
#include "utility.h"
}
#else
#include "utility.h"
#endif

void reversible_boolean_gate(const UINT* target_qubit_index_list, UINT target_qubit_index_count, std::function<ITYPE(ITYPE, ITYPE)> function_ptr, CTYPE* state, ITYPE dim) {

	// matrix dim, mask, buffer
	const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
	const ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);

	// insert index
	const UINT* sorted_insert_index_list = create_sorted_ui_list(target_qubit_index_list, target_qubit_index_count);

	// loop variables
	const ITYPE loop_dim = dim >> target_qubit_index_count;

	CTYPE* buffer = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*matrix_dim));
	ITYPE state_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create base index
		ITYPE basis_0 = state_index;
		for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
			UINT insert_index = sorted_insert_index_list[cursor];
			basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index, insert_index);
		}

		// compute matrix-vector multiply

		for (ITYPE x = 0; x < matrix_dim; ++x) {
			ITYPE y = function_ptr(x, matrix_dim);
			buffer[y] = state[basis_0 ^ matrix_mask_list[x]];
		}

		// set result
		for (ITYPE y = 0; y < matrix_dim; ++y) {
			state[basis_0 ^ matrix_mask_list[y]] = buffer[y];
		}
	}
	free(buffer);
	free((UINT*)sorted_insert_index_list);
	free((ITYPE*)matrix_mask_list);
}
