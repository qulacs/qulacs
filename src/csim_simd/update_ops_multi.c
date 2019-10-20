
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
void multi_qubit_Pauli_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,UINT pivot_qubit_index, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_rotation_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_gate_Z_mask(ITYPE phase_flip_mask, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_rotation_gate_Z_mask(ITYPE phase_flip_mask, double angle, CTYPE* state, ITYPE dim);


void multi_qubit_Pauli_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,UINT pivot_qubit_index, CTYPE* state, ITYPE dim){
    // pivot mask
    const ITYPE pivot_mask = 1ULL << pivot_qubit_index;

    // loop varaibles
    const ITYPE loop_dim = dim/2;
    ITYPE state_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0;state_index<loop_dim;++state_index){

        // create base index
        ITYPE basis_0 = insert_zero_to_basis_index(state_index, pivot_mask, pivot_qubit_index);

        // gather index
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;

        // determine sign
        UINT sign_0 = count_population(basis_0 & phase_flip_mask)%2;
        UINT sign_1 = count_population(basis_1 & phase_flip_mask)%2;
        
        // fetch values
        CTYPE cval_0 = state[basis_0];
        CTYPE cval_1 = state[basis_1];

        // set values
        state[basis_0] = cval_1 * PHASE_M90ROT[(global_phase_90rot_count + sign_0*2)%4];
        state[basis_1] = cval_0 * PHASE_M90ROT[(global_phase_90rot_count + sign_1*2)%4];
    }
}
void multi_qubit_Pauli_rotation_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, CTYPE* state, ITYPE dim){

    // pivot mask
    const ITYPE pivot_mask = 1ULL << pivot_qubit_index;

    // loop varaibles
    const ITYPE loop_dim = dim/2;
    ITYPE state_index;

    // coefs
    const double cosval = cos(angle/2);
    const double sinval = sin(angle/2);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0;state_index<loop_dim;++state_index){

        // create base index
        ITYPE basis_0 = insert_zero_to_basis_index(state_index, pivot_mask, pivot_qubit_index);

        // gather index
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;

        // determine parity
        int bit_parity_0 = count_population(basis_0 & phase_flip_mask)%2;
        int bit_parity_1 = count_population(basis_1 & phase_flip_mask)%2;

        // fetch values        
        CTYPE cval_0 = state[basis_0];
        CTYPE cval_1 = state[basis_1];

        // set values
        state[basis_0] = cosval * cval_0 + 1.i * sinval * cval_1 * PHASE_M90ROT[ (global_phase_90rot_count + bit_parity_0*2)%4 ];
        state[basis_1] = cosval * cval_1 + 1.i * sinval * cval_0 * PHASE_M90ROT[ (global_phase_90rot_count + bit_parity_1*2)%4 ];
    }
}

void multi_qubit_Pauli_gate_Z_mask(ITYPE phase_flip_mask, CTYPE* state, ITYPE dim){

    // loop varaibles
    const ITYPE loop_dim = dim;
    ITYPE state_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0;state_index<loop_dim;++state_index){
        // determine parity
        int bit_parity = count_population(state_index & phase_flip_mask)%2;

        // set values
        if(bit_parity%2==1){
            state[state_index] *= -1;
        }
    }
}


void multi_qubit_Pauli_rotation_gate_Z_mask(ITYPE phase_flip_mask, double angle, CTYPE* state, ITYPE dim){

    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;

    // coefs
    const double cosval = cos(angle/2);
    const double sinval = sin(angle/2);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0;state_index<loop_dim;++state_index){

        // determine sign
        int bit_parity = count_population(state_index & phase_flip_mask)%2;
        int sign = 1 - 2*bit_parity;

        // set value
        state[state_index] *= cosval + (CTYPE)sign * 1.i * sinval;
    }
}

void multi_qubit_Pauli_gate_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, CTYPE* state, ITYPE dim){
    // create pauli mask and call function
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    if(bit_flip_mask == 0){
        multi_qubit_Pauli_gate_Z_mask(phase_flip_mask, state,dim);
    }else{
        multi_qubit_Pauli_gate_XZ_mask(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim);
    }
}

void multi_qubit_Pauli_gate_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, CTYPE* state, ITYPE dim){
    // create pauli mask and call function
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    if(bit_flip_mask == 0){
        multi_qubit_Pauli_gate_Z_mask(phase_flip_mask, state,dim);
    }else{
        multi_qubit_Pauli_gate_XZ_mask(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim);
    }
} 

void multi_qubit_Pauli_rotation_gate_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, double angle, CTYPE* state, ITYPE dim){
    // create pauli mask and call function
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    if(bit_flip_mask == 0){
        multi_qubit_Pauli_rotation_gate_Z_mask(phase_flip_mask, angle, state, dim);
    }else{
        multi_qubit_Pauli_rotation_gate_XZ_mask(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index,angle, state, dim);
    }
}

void multi_qubit_Pauli_rotation_gate_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, double angle, CTYPE* state, ITYPE dim){
    // create pauli mask and call function
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    if(bit_flip_mask == 0){
        multi_qubit_Pauli_rotation_gate_Z_mask(phase_flip_mask, angle, state, dim);
    }else{
        multi_qubit_Pauli_rotation_gate_XZ_mask(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle, state, dim);
    }
} 



void two_qubit_dense_matrix_gate(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim) {

	// target mask
	const UINT target_qubit_index_min = (target_qubit_index1 < target_qubit_index2 ? target_qubit_index1 : target_qubit_index2);
	const UINT target_qubit_index_max = (target_qubit_index1 >= target_qubit_index2 ? target_qubit_index1 : target_qubit_index2);
	const ITYPE target_mask_min = 1ULL << target_qubit_index_min;
	const ITYPE target_mask_max = 1ULL << target_qubit_index_max;
	const ITYPE target_mask1 = 1ULL << target_qubit_index1;
	const ITYPE target_mask2 = 1ULL << target_qubit_index2;

	// loop variables
	const ITYPE loop_dim = dim / 4;
	ITYPE state_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create index
		ITYPE basis_0 = state_index;
		basis_0 = insert_zero_to_basis_index(basis_0, target_mask_min, target_qubit_index_min);
		basis_0 = insert_zero_to_basis_index(basis_0, target_mask_max, target_qubit_index_max);

		// gather index
		ITYPE basis_1 = basis_0 ^ target_mask1;
		ITYPE basis_2 = basis_0 ^ target_mask2;
		ITYPE basis_3 = basis_0 ^ target_mask1 ^ target_mask2;

		// fetch values
		CTYPE cval_0 = state[basis_0];
		CTYPE cval_1 = state[basis_1];
		CTYPE cval_2 = state[basis_2];
		CTYPE cval_3 = state[basis_3];

		// set values
		state[basis_0] = matrix[0] * cval_0 + matrix[1] * cval_1 + matrix[2] * cval_2 + matrix[3] * cval_3;
		state[basis_1] = matrix[4] * cval_0 + matrix[5] * cval_1 + matrix[6] * cval_2 + matrix[7] * cval_3;
		state[basis_2] = matrix[8] * cval_0 + matrix[9] * cval_1 + matrix[10] * cval_2 + matrix[11] * cval_3;
		state[basis_3] = matrix[12] * cval_0 + matrix[13] * cval_1 + matrix[14] * cval_2 + matrix[15] * cval_3;
	}
}


// TODO: malloc should be cached, should not be repeated in every function call.
void multi_qubit_dense_matrix_gate(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim) {

    // matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    const ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);

    // insert index
    const UINT* sorted_insert_index_list = create_sorted_ui_list(target_qubit_index_list, target_qubit_index_count);

    // loop variables
    const ITYPE loop_dim = dim >> target_qubit_index_count;

#ifndef _OPENMP
    CTYPE* buffer = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*matrix_dim));
    ITYPE state_index;
    for(state_index = 0 ; state_index < loop_dim ; ++state_index ){
        // create base index
        ITYPE basis_0 = state_index;
        for(UINT cursor=0; cursor < target_qubit_index_count ; cursor++){
            UINT insert_index = sorted_insert_index_list[cursor];
            basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index , insert_index );
        }

        // compute matrix-vector multiply
        for(ITYPE y = 0 ; y < matrix_dim ; ++y ){
            buffer[y]=0;
            for(ITYPE x = 0 ; x < matrix_dim ; ++x){
                buffer[y] += matrix[y*matrix_dim + x] * state[ basis_0 ^ matrix_mask_list[x] ];
            }
        }

        // set result
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            state[basis_0 ^ matrix_mask_list[y]] = buffer[y];
        }
    }
    free(buffer);
#else
    const UINT thread_count = omp_get_max_threads();
    CTYPE* buffer_list = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*matrix_dim*thread_count));

    const ITYPE block_size = loop_dim / thread_count;
    const ITYPE residual = loop_dim % thread_count;

    #pragma omp parallel
    {
        UINT thread_id = omp_get_thread_num();
        ITYPE start_index = block_size * thread_id + (residual > thread_id ? thread_id : residual);
        ITYPE end_index = block_size * (thread_id + 1) + (residual > (thread_id + 1) ? (thread_id + 1) : residual);
        CTYPE* buffer = buffer_list + thread_id * matrix_dim;

        ITYPE state_index;
        for (state_index = start_index; state_index < end_index; ++state_index) {
            // create base index
            ITYPE basis_0 = state_index;
            for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
                UINT insert_index = sorted_insert_index_list[cursor];
                basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index, insert_index);
            }

            // compute matrix-vector multiply
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
    }
    free(buffer_list);
#endif

    free((UINT*)sorted_insert_index_list);
    free((ITYPE*)matrix_mask_list);
}


void single_qubit_control_multi_qubit_dense_matrix_gate(UINT control_qubit_index, UINT control_value, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim) {

    // matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
    CTYPE* buffer = (CTYPE*)malloc((size_t) (sizeof(CTYPE)*matrix_dim) );

    // insert list
    const UINT insert_index_count = target_qubit_index_count + 1;
    UINT* sorted_insert_index_list = create_sorted_ui_list_value(target_qubit_index_list, target_qubit_index_count ,control_qubit_index);

    // control mask
    const ITYPE control_mask = (1ULL << control_qubit_index) * control_value;

    // loop varaibles
    const ITYPE loop_dim = dim >> insert_index_count;
    ITYPE state_index;


    for(state_index = 0 ; state_index < loop_dim ; ++state_index ){

        // create base index
        ITYPE basis_0 = state_index;
        for(UINT cursor=0; cursor < insert_index_count ; cursor++){
            UINT insert_index = sorted_insert_index_list[cursor];
            basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index , insert_index );
        }

        // flip control
        basis_0 ^= control_mask;

        // compute matrix mul
        for(ITYPE y = 0 ; y < matrix_dim ; ++y ){
            buffer[y]=0;
            for(ITYPE x = 0 ; x < matrix_dim ; ++x){
                buffer[y] += matrix[y*matrix_dim + x] * state[ basis_0 ^ matrix_mask_list[x] ];
            }
        }

        // set result
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            state[basis_0 ^ matrix_mask_list[y]] = buffer[y];
        }
    }
    free(sorted_insert_index_list);
    free(buffer);
    free(matrix_mask_list);
}

void multi_qubit_control_multi_qubit_dense_matrix_gate(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim) {

    // matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
    CTYPE* buffer = (CTYPE*)malloc((size_t) (sizeof(CTYPE)*matrix_dim) );

    // insert index
    const UINT insert_index_count = target_qubit_index_count + control_qubit_index_count;
    UINT* sorted_insert_index_list = create_sorted_ui_list_list(target_qubit_index_list, target_qubit_index_count, control_qubit_index_list, control_qubit_index_count);
    
    // control mask
    ITYPE control_mask = create_control_mask(control_qubit_index_list, control_value_list, control_qubit_index_count);
    
    // loop varaibles
    const ITYPE loop_dim = dim >> (target_qubit_index_count+control_qubit_index_count);
    ITYPE state_index;

    for(state_index = 0 ; state_index < loop_dim ; ++state_index ){

        // create base index
        ITYPE basis_0 = state_index;
        for(UINT cursor=0; cursor < insert_index_count ; cursor++){
            UINT insert_index = sorted_insert_index_list[cursor];
            basis_0 = insert_zero_to_basis_index(basis_0, 1ULL << insert_index , insert_index );
        }

        // flip control masks
        basis_0 ^= control_mask;

        // compute matrix mul
        for(ITYPE y = 0 ; y < matrix_dim ; ++y ){
            buffer[y]=0;
            for(ITYPE x = 0 ; x < matrix_dim ; ++x){
                buffer[y] += matrix[y*matrix_dim+x] * state[ basis_0 ^ matrix_mask_list[x] ];
            }
        }

        // set result
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            state[basis_0 ^ matrix_mask_list[y]] = buffer[y];
        }
    }
    free(sorted_insert_index_list);
    free(buffer);
    free(matrix_mask_list);
}
