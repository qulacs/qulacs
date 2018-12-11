
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

void single_qubit_Pauli_gate(UINT target_qubit_index, UINT Pauli_operator_type, CTYPE *state, ITYPE dim) {
    switch(Pauli_operator_type){
    case 0:
        break;
    case 1:
        X_gate(target_qubit_index,state,dim);
        break;
    case 2:
        Y_gate(target_qubit_index,state,dim);
        break;
    case 3:
        Z_gate(target_qubit_index,state,dim);
        break;
    default:
        fprintf(stderr,"invalid Pauli operation is called");
        assert(0);
    }
}


void single_qubit_Pauli_rotation_gate(UINT target_qubit_index, UINT Pauli_operator_index, double angle, CTYPE *state, ITYPE dim) {
    // create matrix and call dense matrix
    UINT i, j;
    CTYPE rotation_gate[4];
    for(i = 0; i < 2; ++i)
        for(j = 0; j < 2; ++j)
            rotation_gate[i*2+j] = cos(angle/2) * PAULI_MATRIX[0][i*2+j] + sin(angle/2) * 1.0i * PAULI_MATRIX[Pauli_operator_index][i*2+j];

    single_qubit_dense_matrix_gate(target_qubit_index, rotation_gate, state, dim);
}

void single_qubit_dense_matrix_gate(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {

    // target mask
    const ITYPE target_mask = 1ULL << target_qubit_index;

    // loop variables
    const ITYPE loop_dim = dim/2;
    ITYPE state_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim ; ++state_index) {
        // create index
        ITYPE basis_0 = insert_zero_to_basis_index(state_index,target_mask,target_qubit_index);

        // gather index
        ITYPE basis_1 = basis_0 ^ target_mask;

        // fetch values
        CTYPE cval_0 = state[basis_0];
        CTYPE cval_1 = state[basis_1];

        // set values
        state[basis_0] = matrix[0] * cval_0 + matrix[1] * cval_1;
        state[basis_1] = matrix[2] * cval_0 + matrix[3] * cval_1;
    }
}

void single_qubit_diagonal_matrix_gate(UINT target_qubit_index, const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {

    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim ; ++state_index) {
        // determin matrix pos
        UINT bit_val = (state_index >> target_qubit_index)%2;

        // set value
        state[state_index] *= diagonal_matrix[bit_val];
    }
}

void single_qubit_phase_gate(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {

    // target tmask
    const ITYPE mask = 1ULL << target_qubit_index;

    // loop varaibles
    const ITYPE loop_dim = dim/2;
    ITYPE state_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim ; ++state_index) {

        // crate index
        ITYPE basis_1 = insert_zero_to_basis_index(state_index,mask,target_qubit_index) ^ mask;

        // set values
        state[basis_1] *= phase;
    }
}

void single_qubit_control_single_qubit_dense_matrix_gate(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {

    // loop varaibles
    const ITYPE loop_dim = dim>>2;
    ITYPE state_index;

    // target mask
    const ITYPE target_mask = 1ULL << target_qubit_index;
    
    // control mask
    const ITYPE control_mask = (1ULL << control_qubit_index) * control_value;

    // insert index
    const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << max_qubit_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index = 0; state_index < loop_dim; ++state_index) {
        // create base index
        ITYPE basis_c_t0 = state_index;
        basis_c_t0 = insert_zero_to_basis_index(basis_c_t0, min_qubit_mask, min_qubit_index);
        basis_c_t0 = insert_zero_to_basis_index(basis_c_t0 , max_qubit_mask, max_qubit_index);

        // flip control
        basis_c_t0 ^= control_mask;

        // gather index
        ITYPE basis_c_t1 = basis_c_t0 ^ target_mask;

        // fetch values
        CTYPE cval_c_t0 = state[basis_c_t0];
        CTYPE cval_c_t1 = state[basis_c_t1];

        // set values
        state[basis_c_t0] = matrix[0] * cval_c_t0 + matrix[1] * cval_c_t1;
        state[basis_c_t1] = matrix[2] * cval_c_t0 + matrix[3] * cval_c_t1;
    }
}

// This function can be further optimized by disucssing what should be computed before loop as local object.
// Current function is designed to avoid if-statement in loop.
void multi_qubit_control_single_qubit_dense_matrix_gate(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, 
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
    ITYPE state_index;


#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index = 0; state_index < loop_dim; ++state_index) {

        // create base index
        ITYPE basis_c_t0 = state_index;
        for(UINT cursor = 0 ; cursor < insert_index_list_count ; ++cursor){
            basis_c_t0 = insert_zero_to_basis_index(basis_c_t0 , 1ULL << insert_index_list[cursor] , insert_index_list[cursor]);
        }

        // flip controls
        basis_c_t0 ^= control_mask;

        // gather target
        ITYPE basis_c_t1 = basis_c_t0 ^ target_mask;

        // fetch values
        CTYPE cval_c_t0 = state[basis_c_t0];
        CTYPE cval_c_t1 = state[basis_c_t1];

        // set values
        state[basis_c_t0] = matrix[0] * cval_c_t0 + matrix[1] * cval_c_t1;
        state[basis_c_t1] = matrix[2] * cval_c_t0 + matrix[3] * cval_c_t1;
    }

    free(insert_index_list);
}
