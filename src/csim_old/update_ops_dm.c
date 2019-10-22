
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "constant.h"
#include "update_ops_dm.h"
#include "utility.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void dm_normalize(double norm, CTYPE* state, ITYPE dim) {
	const ITYPE loop_dim = dim;
	const double normalize_factor = 1. / norm;
	ITYPE state_index_y;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index_y = 0; state_index_y < loop_dim; ++state_index_y) {
		ITYPE state_index_x;
		for (state_index_x = 0; state_index_x < loop_dim; ++state_index_x) {
			state[state_index_y * dim + state_index_x] *= normalize_factor;
		}
	}
}

void dm_single_qubit_dense_matrix_gate(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {

	// target mask
	const ITYPE target_mask = 1ULL << target_qubit_index;

	// loop variables
	const ITYPE loop_dim = dim / 2;

	// create extended matrix
	CTYPE ext_matrix[16];
	for (int y = 0; y < 4; ++y) {
		int y1 = y / 2;
		int y2 = y % 2;
		for (int x = 0; x < 4; ++x) {
			int x1 = x / 2;
			int x2 = x % 2;
			ext_matrix[y * 4 + x] = matrix[y1 * 2 + x1] * conj(matrix[y2 * 2 + x2]);
		}
	}

	ITYPE state_index_x, state_index_y;
#ifdef _OPENMP
#pragma omp parallel for private(state_index_x)
#endif
	for (state_index_y = 0; state_index_y < loop_dim; ++state_index_y) {

		// create vertical index
		ITYPE basis_0_y = insert_zero_to_basis_index(state_index_y, target_mask, target_qubit_index);
		// flip target bit
		ITYPE basis_1_y = basis_0_y ^ target_mask;

		for (state_index_x = 0; state_index_x < loop_dim; ++state_index_x) {

			// create vertical index
			ITYPE basis_0_x = insert_zero_to_basis_index(state_index_x, target_mask, target_qubit_index);
			// flip target bit
			ITYPE basis_1_x = basis_0_x ^ target_mask;

			ITYPE basis_00 = basis_0_y * dim + basis_0_x;
			ITYPE basis_01 = basis_0_y * dim + basis_1_x;
			ITYPE basis_10 = basis_1_y * dim + basis_0_x;
			ITYPE basis_11 = basis_1_y * dim + basis_1_x;

			// fetch values
			CTYPE cval_00 = state[basis_00];
			CTYPE cval_01 = state[basis_01];
			CTYPE cval_10 = state[basis_10];
			CTYPE cval_11 = state[basis_11];

			// set values
			state[basis_00] = ext_matrix[0] * cval_00 + ext_matrix[1] * cval_01 + ext_matrix[2] * cval_10 + ext_matrix[3] * cval_11;
			state[basis_01] = ext_matrix[4] * cval_00 + ext_matrix[5] * cval_01 + ext_matrix[6] * cval_10 + ext_matrix[7] * cval_11;
			state[basis_10] = ext_matrix[8] * cval_00 + ext_matrix[9] * cval_01 + ext_matrix[10] * cval_10 + ext_matrix[11] * cval_11;
			state[basis_11] = ext_matrix[12] * cval_00 + ext_matrix[13] * cval_01 + ext_matrix[14] * cval_10 + ext_matrix[15] * cval_11;
		}
	}
}


void dm_multi_qubit_control_single_qubit_dense_matrix_gate(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count,
	UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {

	// insert index list
	const UINT insert_index_list_count = control_qubit_index_count + 1;
	UINT* insert_index_list = create_sorted_ui_list_value(control_qubit_index_list, control_qubit_index_count, target_qubit_index);

	// target mask
	const ITYPE target_mask = 1ULL << target_qubit_index;

	// control mask
	ITYPE control_mask = create_control_mask(control_qubit_index_list, control_value_list, control_qubit_index_count);

	// loop variables
	const ITYPE loop_dim = dim >> insert_index_list_count;

	CTYPE adjoint_matrix[4];
	adjoint_matrix[0] = conj(matrix[0]);
	adjoint_matrix[1] = conj(matrix[2]);
	adjoint_matrix[2] = conj(matrix[1]);
	adjoint_matrix[3] = conj(matrix[3]);

	ITYPE state_index_x, state_index_y;
#ifdef _OPENMP
#pragma omp parallel for private(state_index_y)
#endif
	for (state_index_x = 0; state_index_x < dim; ++state_index_x) {

		for (state_index_y = 0; state_index_y < loop_dim; ++state_index_y) {

			// create base index
			ITYPE basis_c_t0_y = state_index_y;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_c_t0_y = insert_zero_to_basis_index(basis_c_t0_y, 1ULL << insert_index_list[cursor], insert_index_list[cursor]);
			}

			// flip controls
			basis_c_t0_y ^= control_mask;

			// gather target
			ITYPE basis_c_t1_y = basis_c_t0_y ^ target_mask;

			// set index
			ITYPE basis_0 = basis_c_t0_y * dim + state_index_x;
			ITYPE basis_1 = basis_c_t1_y * dim + state_index_x;

			// fetch values
			CTYPE cval_0 = state[basis_0];
			CTYPE cval_1 = state[basis_1];

			// set values
			state[basis_0] = matrix[0] * cval_0 + matrix[1] * cval_1;
			state[basis_1] = matrix[2] * cval_0 + matrix[3] * cval_1;
		}
	}

#ifdef _OPENMP
#pragma omp parallel for private(state_index_x)
#endif
	for (state_index_y = 0; state_index_y < dim; ++state_index_y) {

		for (state_index_x = 0; state_index_x < loop_dim; ++state_index_x) {

			// create base index
			ITYPE basis_c_t0_x = state_index_x;
			for (UINT cursor = 0; cursor < insert_index_list_count; ++cursor) {
				basis_c_t0_x = insert_zero_to_basis_index(basis_c_t0_x, 1ULL << insert_index_list[cursor], insert_index_list[cursor]);
			}

			// flip controls
			basis_c_t0_x ^= control_mask;

			// gather target
			ITYPE basis_c_t1_x = basis_c_t0_x ^ target_mask;

			// set index
			ITYPE basis_0 = state_index_y * dim + basis_c_t0_x;
			ITYPE basis_1 = state_index_y * dim + basis_c_t1_x;

			// fetch values
			CTYPE cval_0 = state[basis_0];
			CTYPE cval_1 = state[basis_1];

			// set values
			state[basis_0] = cval_0 * adjoint_matrix[0] + cval_1 * adjoint_matrix[2];
			state[basis_1] = cval_0 * adjoint_matrix[1] + cval_1 * adjoint_matrix[3];
		}
	}

	free(insert_index_list);
}

/*
// inefficient implementation
void dm_multi_qubit_dense_matrix_gate(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim) {

	// matrix dim, mask, buffer
	const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
	const ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);

	// create extended matrix
	const ITYPE ext_matrix_dim = matrix_dim*matrix_dim;
	CTYPE* ext_matrix = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*(ext_matrix_dim * ext_matrix_dim)));
	for (ITYPE y = 0; y < ext_matrix_dim; ++y) {
		ITYPE y1 = y / matrix_dim;
		ITYPE y2 = y % matrix_dim;
		for (ITYPE x = 0; x < ext_matrix_dim; ++x) {
			ITYPE x1 = x / matrix_dim;
			ITYPE x2 = x % matrix_dim;
			ext_matrix[y*ext_matrix_dim + x] = matrix[y1*matrix_dim + x1] * conj(matrix[y2*matrix_dim + x2]);
		}
	}

	// insert index
	const UINT* sorted_insert_index_list = create_sorted_ui_list(target_qubit_index_list, target_qubit_index_count);

	// loop variables
	const ITYPE loop_dim = dim >> target_qubit_index_count;

#ifndef _OPENMP
	CTYPE* buffer = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*ext_matrix_dim));
	ITYPE state_index_y;
	for (state_index_y = 0; state_index_y < loop_dim; ++state_index_y) {
		// create base index
		ITYPE basis_0_y = state_index_y;
		for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
			UINT insert_index = sorted_insert_index_list[cursor];
			basis_0_y = insert_zero_to_basis_index(basis_0_y, 1ULL << insert_index, insert_index);
		}

		ITYPE state_index_x;
		for (state_index_x = 0; state_index_x < loop_dim; ++state_index_x) {
			// create base index
			ITYPE basis_0_x = state_index_x;
			for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
				UINT insert_index = sorted_insert_index_list[cursor];
				basis_0_x = insert_zero_to_basis_index(basis_0_x, 1ULL << insert_index, insert_index);
			}

			// compute matrix-vector multiply
			for (ITYPE y = 0; y < ext_matrix_dim; ++y) {
				buffer[y] = 0;
				for (ITYPE x = 0; x < ext_matrix_dim; ++x) {
					ITYPE dm_index_x = basis_0_x ^ matrix_mask_list[x%matrix_dim];
					ITYPE dm_index_y = basis_0_y ^ matrix_mask_list[x/matrix_dim];
					buffer[y] += ext_matrix[y*ext_matrix_dim + x] * state[ dm_index_y * dim + dm_index_x];
				}
			}

			// set result
			for (ITYPE y = 0; y < ext_matrix_dim; ++y) {
				ITYPE dm_index_x = basis_0_x ^ matrix_mask_list[y % matrix_dim];
				ITYPE dm_index_y = basis_0_y ^ matrix_mask_list[y / matrix_dim];
				state[dm_index_y * dim + dm_index_x] = buffer[y];
			}
		}
	}
	free(buffer);
#else
	const UINT thread_count = omp_get_max_threads();
	CTYPE* buffer_list = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*ext_matrix_dim*thread_count));

	const ITYPE block_size = loop_dim / thread_count;
	const ITYPE residual = loop_dim % thread_count;

#pragma omp parallel
	{
		UINT thread_id = omp_get_thread_num();
		ITYPE start_index = block_size * thread_id + (residual > thread_id ? thread_id : residual);
		ITYPE end_index = block_size * (thread_id + 1) + (residual > (thread_id + 1) ? (thread_id + 1) : residual);
		CTYPE* buffer = buffer_list + thread_id * ext_matrix_dim;

		ITYPE state_index_y;
		for (state_index_y = start_index; state_index_y < end_index; ++state_index_y) {

			// create base index
			ITYPE basis_0_y = state_index_y;
			for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
				UINT insert_index = sorted_insert_index_list[cursor];
				basis_0_y = insert_zero_to_basis_index(basis_0_y, 1ULL << insert_index, insert_index);
			}

			ITYPE state_index_x;
			for (state_index_x = 0; state_index_x < loop_dim; ++state_index_x) {
				// create base index
				ITYPE basis_0_x = state_index_x;
				for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
					UINT insert_index = sorted_insert_index_list[cursor];
					basis_0_x = insert_zero_to_basis_index(basis_0_x, 1ULL << insert_index, insert_index);
				}

				// compute matrix-vector multiply
				for (ITYPE y = 0; y < ext_matrix_dim; ++y) {
					buffer[y] = 0;
					for (ITYPE x = 0; x < ext_matrix_dim; ++x) {
						ITYPE dm_index_x = basis_0_x ^ matrix_mask_list[x%matrix_dim];
						ITYPE dm_index_y = basis_0_y ^ matrix_mask_list[x / matrix_dim];
						buffer[y] += ext_matrix[y*ext_matrix_dim + x] * state[dm_index_y * dim + dm_index_x];
					}
				}

				// set result
				for (ITYPE y = 0; y < ext_matrix_dim; ++y) {
					ITYPE dm_index_x = basis_0_x ^ matrix_mask_list[y % matrix_dim];
					ITYPE dm_index_y = basis_0_y ^ matrix_mask_list[y / matrix_dim];
					state[dm_index_y * dim + dm_index_x] = buffer[y];
				}
			}
		}
	}
	free(buffer_list);
#endif
	free(ext_matrix);
	free((UINT*)sorted_insert_index_list);
	free((ITYPE*)matrix_mask_list);
}
*/


void dm_multi_qubit_dense_matrix_gate(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim) {

	// matrix dim, mask, buffer
	const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
	const ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);

	// create extended matrix
	CTYPE* adjoint_matrix = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*(matrix_dim * matrix_dim)));
	for (ITYPE y = 0; y < matrix_dim; ++y) {
		for (ITYPE x = 0; x < matrix_dim; ++x) {
			adjoint_matrix[y*matrix_dim + x] = conj(matrix[x*matrix_dim + y]);
		}
	}

	// insert index
	const UINT* sorted_insert_index_list = create_sorted_ui_list(target_qubit_index_list, target_qubit_index_count);

	// loop variables
	const ITYPE loop_dim = dim >> target_qubit_index_count;

#ifndef _OPENMP
	CTYPE* buffer = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*matrix_dim*matrix_dim));
	ITYPE state_index_y;
	for (state_index_y = 0; state_index_y < loop_dim; ++state_index_y) {
		// create base index
		ITYPE basis_0_y = state_index_y;
		for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
			UINT insert_index = sorted_insert_index_list[cursor];
			basis_0_y = insert_zero_to_basis_index(basis_0_y, 1ULL << insert_index, insert_index);
		}

		ITYPE state_index_x;
		for (state_index_x = 0; state_index_x < loop_dim; ++state_index_x) {
			// create base index
			ITYPE basis_0_x = state_index_x;
			for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
				UINT insert_index = sorted_insert_index_list[cursor];
				basis_0_x = insert_zero_to_basis_index(basis_0_x, 1ULL << insert_index, insert_index);
			}

			// compute matrix-matrix multiply
			// TODO: improve matmul
			for (ITYPE y = 0; y < matrix_dim; ++y) {
				for (ITYPE x = 0; x < matrix_dim; ++x) {
					buffer[y*matrix_dim + x] = 0;
					for (ITYPE k = 0; k < matrix_dim; ++k) {
						ITYPE dm_index_x = basis_0_x ^ matrix_mask_list[x];
						ITYPE dm_index_k = basis_0_y ^ matrix_mask_list[k];
						buffer[y*matrix_dim+x] += matrix[y*matrix_dim + k] * state[ dm_index_k * dim + dm_index_x];
					}
				}
			}

			for (ITYPE y = 0; y < matrix_dim; ++y) {
				for (ITYPE x = 0; x < matrix_dim; ++x) {
					ITYPE dm_index_x = basis_0_x ^ matrix_mask_list[x];
					ITYPE dm_index_y = basis_0_y ^ matrix_mask_list[y];
					ITYPE dm_index = dm_index_y * dim + dm_index_x;
					state[dm_index] = 0;
					for (ITYPE k = 0; k < matrix_dim; ++k) {
						state[dm_index] += buffer[y*matrix_dim + k] * adjoint_matrix[k*matrix_dim + x];
					}
				}
			}
		}
	}
	free(buffer);
#else
	const UINT thread_count = omp_get_max_threads();
	CTYPE* buffer_list = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*matrix_dim*matrix_dim*thread_count));

	const ITYPE block_size = loop_dim / thread_count;
	const ITYPE residual = loop_dim % thread_count;

#pragma omp parallel
	{
		UINT thread_id = omp_get_thread_num();
		ITYPE start_index = block_size * thread_id + (residual > thread_id ? thread_id : residual);
		ITYPE end_index = block_size * (thread_id + 1) + (residual > (thread_id + 1) ? (thread_id + 1) : residual);
		CTYPE* buffer = buffer_list + thread_id * matrix_dim*matrix_dim;

		ITYPE state_index_y;
		for (state_index_y = start_index; state_index_y < end_index; ++state_index_y) {

			// create base index
			ITYPE basis_0_y = state_index_y;
			for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
				UINT insert_index = sorted_insert_index_list[cursor];
				basis_0_y = insert_zero_to_basis_index(basis_0_y, 1ULL << insert_index, insert_index);
			}

			ITYPE state_index_x;
			for (state_index_x = 0; state_index_x < loop_dim; ++state_index_x) {
				// create base index
				ITYPE basis_0_x = state_index_x;
				for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
					UINT insert_index = sorted_insert_index_list[cursor];
					basis_0_x = insert_zero_to_basis_index(basis_0_x, 1ULL << insert_index, insert_index);
				}

				// compute matrix-matrix multiply
				// TODO: improve matmul
				for (ITYPE y = 0; y < matrix_dim; ++y) {
					for (ITYPE x = 0; x < matrix_dim; ++x) {
						buffer[y*matrix_dim + x] = 0;
						for (ITYPE k = 0; k < matrix_dim; ++k) {
							ITYPE dm_index_x = basis_0_x ^ matrix_mask_list[x];
							ITYPE dm_index_k = basis_0_y ^ matrix_mask_list[k];
							buffer[y*matrix_dim + x] += matrix[y*matrix_dim + k] * state[dm_index_k * dim + dm_index_x];
						}
					}
				}

				for (ITYPE y = 0; y < matrix_dim; ++y) {
					for (ITYPE x = 0; x < matrix_dim; ++x) {
						ITYPE dm_index_x = basis_0_x ^ matrix_mask_list[x];
						ITYPE dm_index_y = basis_0_y ^ matrix_mask_list[y];
						ITYPE dm_index = dm_index_y * dim + dm_index_x;
						state[dm_index] = 0;
						for (ITYPE k = 0; k < matrix_dim; ++k) {
							state[dm_index] += buffer[y*matrix_dim + k] * adjoint_matrix[k*matrix_dim + x];
						}
					}
				}
			}
		}
	}
	free(buffer_list);
#endif
	free(adjoint_matrix);
	free((UINT*)sorted_insert_index_list);
	free((ITYPE*)matrix_mask_list);
}


void dm_multi_qubit_control_multi_qubit_dense_matrix_gate(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim) {

	// matrix dim, mask, buffer
	const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
	ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);

	// insert index
	const UINT insert_index_count = target_qubit_index_count + control_qubit_index_count;
	UINT* sorted_insert_index_list = create_sorted_ui_list_list(target_qubit_index_list, target_qubit_index_count, control_qubit_index_list, control_qubit_index_count);

	// control mask
	ITYPE control_mask = create_control_mask(control_qubit_index_list, control_value_list, control_qubit_index_count);

	// loop varaibles
	const ITYPE loop_dim = dim >> (target_qubit_index_count + control_qubit_index_count);

	CTYPE* adjoint_matrix = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*matrix_dim*matrix_dim));
	for (ITYPE y = 0; y < matrix_dim; ++y) {
		for (ITYPE x = 0; x < matrix_dim; ++x) {
			adjoint_matrix[y*matrix_dim + x] = conj(matrix[x*matrix_dim + y]);
		}
	}

#ifndef _OPENMP
	CTYPE* buffer = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*matrix_dim));
	ITYPE state_index_x, state_index_y;
	for (state_index_x = 0; state_index_x < dim; ++state_index_x) {
		for (state_index_y = 0; state_index_y < loop_dim; ++state_index_y) {
			// create base index
			ITYPE basis_0_y = state_index_y;
			for (UINT cursor = 0; cursor < insert_index_count; cursor++) {
				UINT insert_index = sorted_insert_index_list[cursor];
				basis_0_y = insert_zero_to_basis_index(basis_0_y, 1ULL << insert_index, insert_index);
			}

			// flip control masks
			basis_0_y ^= control_mask;

			// compute matrix vector mul
			for (ITYPE y = 0; y < matrix_dim; ++y) {
				buffer[y] = 0;
				for (ITYPE x = 0; x < matrix_dim; ++x) {
					ITYPE dm_index_y = basis_0_y ^ matrix_mask_list[x];
					buffer[y] += matrix[y*matrix_dim + x] * state[dm_index_y*dim + state_index_x];
				}
			}

			// set result
			for (ITYPE y = 0; y < matrix_dim; ++y) {
				ITYPE dm_index_y = basis_0_y ^ matrix_mask_list[y];
				state[dm_index_y*dim + state_index_x] = buffer[y];
			}
		}
	}
	for (state_index_y = 0; state_index_y < dim; ++state_index_y) {
		for (state_index_x = 0; state_index_x < loop_dim; ++state_index_x) {
			// create base index
			ITYPE basis_0_x = state_index_x;
			for (UINT cursor = 0; cursor < insert_index_count; cursor++) {
				UINT insert_index = sorted_insert_index_list[cursor];
				basis_0_x = insert_zero_to_basis_index(basis_0_x, 1ULL << insert_index, insert_index);
			}

			// flip control masks
			basis_0_x ^= control_mask;

			// compute matrix vector mul
			for (ITYPE y = 0; y < matrix_dim; ++y) {
				buffer[y] = 0;
				for (ITYPE x = 0; x < matrix_dim; ++x) {
					ITYPE dm_index_x = basis_0_x ^ matrix_mask_list[x];
					buffer[y] += state[state_index_y*dim + dm_index_x] * adjoint_matrix[x*matrix_dim + y];
				}
			}

			// set result
			for (ITYPE y = 0; y < matrix_dim; ++y) {
				ITYPE dm_index_x = basis_0_x ^ matrix_mask_list[y];
				state[state_index_y*dim + dm_index_x] = buffer[y];
			}
		}
	}
	free(buffer);
#else
	const UINT thread_count = omp_get_max_threads();
	CTYPE* buffer_list = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*matrix_dim*thread_count));
	const ITYPE block_size = dim / thread_count;
	const ITYPE residual = dim % thread_count;
#pragma omp parallel
	{
		UINT thread_id = omp_get_thread_num();
		ITYPE start_index = block_size * thread_id + (residual > thread_id ? thread_id : residual);
		ITYPE end_index = block_size * (thread_id + 1) + (residual > (thread_id + 1) ? (thread_id + 1) : residual);
		CTYPE* buffer = buffer_list + thread_id * matrix_dim;

		ITYPE state_index_y, state_index_x;
		for (state_index_x = start_index; state_index_x < end_index; ++state_index_x) {
			for (state_index_y = 0; state_index_y < loop_dim; ++state_index_y) {
				// create base index
				ITYPE basis_0_y = state_index_y;
				for (UINT cursor = 0; cursor < insert_index_count; cursor++) {
					UINT insert_index = sorted_insert_index_list[cursor];
					basis_0_y = insert_zero_to_basis_index(basis_0_y, 1ULL << insert_index, insert_index);
				}

				// flip control masks
				basis_0_y ^= control_mask;

				// compute matrix vector mul
				for (ITYPE y = 0; y < matrix_dim; ++y) {
					buffer[y] = 0;
					for (ITYPE x = 0; x < matrix_dim; ++x) {
						ITYPE dm_index_y = basis_0_y ^ matrix_mask_list[x];
						buffer[y] += matrix[y*matrix_dim + x] * state[dm_index_y*dim + state_index_x];
					}
				}

				// set result
				for (ITYPE y = 0; y < matrix_dim; ++y) {
					ITYPE dm_index_y = basis_0_y ^ matrix_mask_list[y];
					state[dm_index_y*dim + state_index_x] = buffer[y];
				}
			}
		}
#pragma omp barrier
		for (state_index_y = start_index; state_index_y < end_index; ++state_index_y) {
			for (state_index_x = 0; state_index_x < loop_dim; ++state_index_x) {
				// create base index
				ITYPE basis_0_x = state_index_x;
				for (UINT cursor = 0; cursor < insert_index_count; cursor++) {
					UINT insert_index = sorted_insert_index_list[cursor];
					basis_0_x = insert_zero_to_basis_index(basis_0_x, 1ULL << insert_index, insert_index);
				}

				// flip control masks
				basis_0_x ^= control_mask;

				// compute matrix vector mul
				for (ITYPE y = 0; y < matrix_dim; ++y) {
					buffer[y] = 0;
					for (ITYPE x = 0; x < matrix_dim; ++x) {
						ITYPE dm_index_x = basis_0_x ^ matrix_mask_list[x];
						buffer[y] += state[state_index_y*dim + dm_index_x] * adjoint_matrix[x*matrix_dim + y];
					}
				}

				// set result
				for (ITYPE y = 0; y < matrix_dim; ++y) {
					ITYPE dm_index_x = basis_0_x ^ matrix_mask_list[y];
					state[state_index_y*dim + dm_index_x] = buffer[y];
				}
			}
		}
	}
	free(buffer_list); 
#endif
	free(adjoint_matrix);
	free(sorted_insert_index_list);
	free(matrix_mask_list);
}


void dm_X_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, PAULI_MATRIX[1], state, dim);
}
void dm_Y_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, PAULI_MATRIX[2], state, dim);
}
void dm_Z_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, PAULI_MATRIX[3], state, dim);
}
void dm_S_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, S_GATE_MATRIX, state, dim);
}
void dm_Sdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, S_DAG_GATE_MATRIX, state, dim);
}
void dm_T_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
	dm_single_qubit_dense_matrix_gate(target_qubit_index, T_GATE_MATRIX, state, dim);
}
void dm_Tdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, T_DAG_GATE_MATRIX, state, dim);
}
void dm_sqrtX_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, SQRT_X_GATE_MATRIX, state, dim);
}
void dm_sqrtXdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, SQRT_X_DAG_GATE_MATRIX, state, dim);
}
void dm_sqrtY_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, SQRT_Y_GATE_MATRIX, state, dim);
}
void dm_sqrtYdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, SQRT_Y_DAG_GATE_MATRIX, state, dim);
}
void dm_H_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, HADAMARD_MATRIX, state, dim);
}
void dm_P0_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, PROJ_0_MATRIX, state, dim);
}
void dm_P1_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	dm_single_qubit_dense_matrix_gate(target_qubit_index, PROJ_1_MATRIX, state, dim);
}
void dm_CNOT_gate(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	UINT control_index_list[1];
	UINT control_value_list[1];
	control_index_list[0] = control_qubit_index;
	control_value_list[0] = 1;
	dm_multi_qubit_control_single_qubit_dense_matrix_gate(control_index_list, control_value_list, 1, target_qubit_index, PAULI_MATRIX[1], state, dim);
}
void dm_CZ_gate(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	UINT control_index_list[1];
	UINT control_value_list[1];
	control_index_list[0] = control_qubit_index;
	control_value_list[0] = 1;
	dm_multi_qubit_control_single_qubit_dense_matrix_gate(control_index_list, control_value_list, 1, target_qubit_index, PAULI_MATRIX[3], state, dim);
}
void dm_SWAP_gate(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim) {
	CTYPE matrix[16];
	memset(matrix, 0, sizeof(CTYPE) * 16);
	matrix[0 * 4 + 0] = 1;
	matrix[1 * 4 + 2] = 1;
	matrix[2 * 4 + 1] = 1;
	matrix[3 * 4 + 3] = 1;
	UINT target_index[2];
	target_index[0] = target_qubit_index_0;
	target_index[1] = target_qubit_index_1;
	dm_multi_qubit_dense_matrix_gate(target_index, 2, matrix, state, dim);
}
void dm_RX_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim) {
	UINT i, j;
	CTYPE rotation_gate[4];
	for (i = 0; i < 2; ++i)
		for (j = 0; j < 2; ++j)
			rotation_gate[i * 2 + j] = cos(angle / 2) * PAULI_MATRIX[0][i * 2 + j] + sin(angle / 2) * 1.0i * PAULI_MATRIX[1][i * 2 + j];
	dm_single_qubit_dense_matrix_gate(target_qubit_index, rotation_gate, state, dim);
}
void dm_RY_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim) {
	UINT i, j;
	CTYPE rotation_gate[4];
	for (i = 0; i < 2; ++i)
		for (j = 0; j < 2; ++j)
			rotation_gate[i * 2 + j] = cos(angle / 2) * PAULI_MATRIX[0][i * 2 + j] + sin(angle / 2) * 1.0i * PAULI_MATRIX[2][i * 2 + j];
	dm_single_qubit_dense_matrix_gate(target_qubit_index, rotation_gate, state, dim);
}
void dm_RZ_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim) {
	UINT i, j;
	CTYPE rotation_gate[4];
	for (i = 0; i < 2; ++i)
		for (j = 0; j < 2; ++j)
			rotation_gate[i * 2 + j] = cos(angle / 2) * PAULI_MATRIX[0][i * 2 + j] + sin(angle / 2) * 1.0i * PAULI_MATRIX[3][i * 2 + j];
	dm_single_qubit_dense_matrix_gate(target_qubit_index, rotation_gate, state, dim);
}


void dm_multi_qubit_Pauli_gate_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, CTYPE* state, ITYPE dim) {
	// TODO faster impl
	const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
	CTYPE* matrix = (CTYPE*)malloc(sizeof(CTYPE)*matrix_dim*matrix_dim);
	for (ITYPE y = 0; y < matrix_dim; ++y) {
		for (ITYPE x = 0; x < matrix_dim; ++x) {
			CTYPE coef = 1.0;
			for (UINT i = 0; i < target_qubit_index_count; ++i) {
				UINT xi = (x >> i) % 2;
				UINT yi = (y >> i) % 2;
				coef *= PAULI_MATRIX[Pauli_operator_type_list[i]][yi*2+xi];
			}
			matrix[y*matrix_dim + x] = coef;
		}
	}
	dm_multi_qubit_dense_matrix_gate(target_qubit_index_list, target_qubit_index_count, matrix, state, dim);
	free(matrix);
}
void dm_multi_qubit_Pauli_rotation_gate_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, double angle, CTYPE* state, ITYPE dim) {
	// TODO faster impl
	const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
	CTYPE* matrix = (CTYPE*)malloc(sizeof(CTYPE)*matrix_dim*matrix_dim);
	for (ITYPE y = 0; y < matrix_dim; ++y) {
		for (ITYPE x = 0; x < matrix_dim; ++x) {
			CTYPE coef = 1.0;
			for (UINT i = 0; i < target_qubit_index_count; ++i) {
				UINT xi = (x >> i) % 2;
				UINT yi = (y >> i) % 2;
				coef *= PAULI_MATRIX[Pauli_operator_type_list[i]][yi*2+xi];
			}
			if (y == x) {
				matrix[y*matrix_dim + x] = cos(angle / 2) *1.0 + 1.0i * sin(angle / 2)*coef;
			}
			else {
				matrix[y*matrix_dim + x] = 1.0i * sin(angle / 2)*coef;
			}
		}
	}
	dm_multi_qubit_dense_matrix_gate(target_qubit_index_list, target_qubit_index_count, matrix, state, dim);
	free(matrix);
}

