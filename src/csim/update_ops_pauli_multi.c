
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

/**
 * perform multi_qubit_Pauli_gate with XZ mask.
 *
 * This function assumes bit_flip_mask is not 0, i.e., at least one bit is flipped. If no bit is flipped, use multi_qubit_Pauli_gate_Z_mask.
 * This function update the quantum state with Pauli operation.
 * bit_flip_mask, phase_flip_mask, global_phase_90rot_count, and pivot_qubit_index must be computed before calling this function.
 * See get_masks_from_*_list for the above four arguemnts.
 */
void multi_qubit_Pauli_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_rotation_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_gate_Z_mask(ITYPE phase_flip_mask, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_rotation_gate_Z_mask(ITYPE phase_flip_mask, double angle, CTYPE* state, ITYPE dim);


void multi_qubit_Pauli_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, CTYPE* state, ITYPE dim) {
	// pivot mask
	const ITYPE pivot_mask = 1ULL << pivot_qubit_index;

	// loop varaibles
	const ITYPE loop_dim = dim / 2;
	ITYPE state_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; ++state_index) {

		// create base index
		ITYPE basis_0 = insert_zero_to_basis_index(state_index, pivot_mask, pivot_qubit_index);

		// gather index
		ITYPE basis_1 = basis_0 ^ bit_flip_mask;

		// determine sign
		UINT sign_0 = count_population(basis_0 & phase_flip_mask) % 2;
		UINT sign_1 = count_population(basis_1 & phase_flip_mask) % 2;

		// fetch values
		CTYPE cval_0 = state[basis_0];
		CTYPE cval_1 = state[basis_1];

		// set values
		state[basis_0] = cval_1 * PHASE_M90ROT[(global_phase_90rot_count + sign_0 * 2) % 4];
		state[basis_1] = cval_0 * PHASE_M90ROT[(global_phase_90rot_count + sign_1 * 2) % 4];
	}
}
void multi_qubit_Pauli_rotation_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, CTYPE* state, ITYPE dim) {

	// pivot mask
	const ITYPE pivot_mask = 1ULL << pivot_qubit_index;

	// loop varaibles
	const ITYPE loop_dim = dim / 2;
	ITYPE state_index;

	// coefs
	const double cosval = cos(angle / 2);
	const double sinval = sin(angle / 2);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; ++state_index) {

		// create base index
		ITYPE basis_0 = insert_zero_to_basis_index(state_index, pivot_mask, pivot_qubit_index);

		// gather index
		ITYPE basis_1 = basis_0 ^ bit_flip_mask;

		// determine parity
		int bit_parity_0 = count_population(basis_0 & phase_flip_mask) % 2;
		int bit_parity_1 = count_population(basis_1 & phase_flip_mask) % 2;

		// fetch values        
		CTYPE cval_0 = state[basis_0];
		CTYPE cval_1 = state[basis_1];

		// set values
		state[basis_0] = cosval * cval_0 + 1.i * sinval * cval_1 * PHASE_M90ROT[(global_phase_90rot_count + bit_parity_0 * 2) % 4];
		state[basis_1] = cosval * cval_1 + 1.i * sinval * cval_0 * PHASE_M90ROT[(global_phase_90rot_count + bit_parity_1 * 2) % 4];
	}
}

void multi_qubit_Pauli_gate_Z_mask(ITYPE phase_flip_mask, CTYPE* state, ITYPE dim) {

	// loop varaibles
	const ITYPE loop_dim = dim;
	ITYPE state_index;

#ifdef _OPENMP
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
}


void multi_qubit_Pauli_rotation_gate_Z_mask(ITYPE phase_flip_mask, double angle, CTYPE* state, ITYPE dim) {

	// loop variables
	const ITYPE loop_dim = dim;
	ITYPE state_index;

	// coefs
	const double cosval = cos(angle / 2);
	const double sinval = sin(angle / 2);

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; ++state_index) {

		// determine sign
		int bit_parity = count_population(state_index & phase_flip_mask) % 2;
		int sign = 1 - 2 * bit_parity;

		// set value
		state[state_index] *= cosval + (CTYPE)sign * 1.i * sinval;
	}
}

void multi_qubit_Pauli_gate_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, CTYPE* state, ITYPE dim) {
	// create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_partial_list(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if (bit_flip_mask == 0) {
		multi_qubit_Pauli_gate_Z_mask(phase_flip_mask, state, dim);
	}
	else {
		multi_qubit_Pauli_gate_XZ_mask(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim);
	}
}

void multi_qubit_Pauli_gate_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, CTYPE* state, ITYPE dim) {
	// create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if (bit_flip_mask == 0) {
		multi_qubit_Pauli_gate_Z_mask(phase_flip_mask, state, dim);
	}
	else {
		multi_qubit_Pauli_gate_XZ_mask(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim);
	}
}

void multi_qubit_Pauli_rotation_gate_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, double angle, CTYPE* state, ITYPE dim) {
	// create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_partial_list(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if (bit_flip_mask == 0) {
		multi_qubit_Pauli_rotation_gate_Z_mask(phase_flip_mask, angle, state, dim);
	}
	else {
		multi_qubit_Pauli_rotation_gate_XZ_mask(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle, state, dim);
	}
}

void multi_qubit_Pauli_rotation_gate_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, double angle, CTYPE* state, ITYPE dim) {
	// create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if (bit_flip_mask == 0) {
		multi_qubit_Pauli_rotation_gate_Z_mask(phase_flip_mask, angle, state, dim);
	}
	else {
		multi_qubit_Pauli_rotation_gate_XZ_mask(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle, state, dim);
	}
}


