
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
