
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "constant.h"
#include "update_ops.h"
#include "utility.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void X_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    const ITYPE mask = (1ULL << target_qubit_index);
    ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0 ; state_index<loop_dim ; ++state_index){
        ITYPE basis_index_0 = insert_zero_to_basis_index(state_index,mask,target_qubit_index);
        ITYPE basis_index_1 = basis_index_0 ^ mask;
        swap_amplitude(state,basis_index_0,basis_index_1);
    }
}

void Y_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    const ITYPE mask = (1ULL << target_qubit_index);
    ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0 ; state_index<loop_dim ; ++state_index){
        ITYPE basis_index_0 = insert_zero_to_basis_index(state_index,mask,target_qubit_index);
        ITYPE basis_index_1 = basis_index_0 ^ mask;
        CTYPE cval_0 = state[basis_index_0];
        CTYPE cval_1 = state[basis_index_1];
        state[basis_index_0] = -cval_1 * 1.i;
        state[basis_index_1] = cval_0 * 1.i;
    }
}

void Z_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    ITYPE state_index;
    ITYPE mask = (1ULL << target_qubit_index);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0 ; state_index<loop_dim ; ++state_index){
        ITYPE temp_index = insert_zero_to_basis_index(state_index,mask,target_qubit_index) ^ mask;
        state[temp_index] *= -1;
    }
}

/** Hadamard gate  **/
void H_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
    single_qubit_dense_matrix_gate(target_qubit_index, HADAMARD_MATRIX, state, dim);
}

void CNOT_gate(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim){
    const ITYPE loop_dim = dim/4;
    const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << max_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;
    const ITYPE target_mask = 1ULL << target_qubit_index;
    ITYPE state_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_insert_only_min = insert_zero_to_basis_index(state_index, min_qubit_mask, min_qubit_index);
        ITYPE basis_c1t0 = insert_zero_to_basis_index( basis_insert_only_min , max_qubit_mask, max_qubit_index) ^ control_mask;
        ITYPE basis_c1t1 = basis_c1t0 ^ target_mask;
        swap_amplitude(state,basis_c1t0,basis_c1t1);
    }
}


void CZ_gate(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim/4;
    const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << max_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;
    const ITYPE target_mask = 1ULL << target_qubit_index;
    ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_insert_only_min = insert_zero_to_basis_index(state_index,min_qubit_mask,min_qubit_index);
        ITYPE basis_c1t1 = insert_zero_to_basis_index( basis_insert_only_min,max_qubit_mask,max_qubit_index) ^ control_mask ^ target_mask;
        state[basis_c1t1] *= -1;
    }
}

void SWAP_gate(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim/4;
    const UINT min_qubit_index = get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index = get_max_ui(target_qubit_index_0, target_qubit_index_1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << max_qubit_index;
    const ITYPE target_mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE target_mask_1 = 1ULL << target_qubit_index_1;
    ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index = 0; state_index < loop_dim ; ++state_index) {
        ITYPE basis_insert_only_min = insert_zero_to_basis_index(state_index, min_qubit_mask, min_qubit_index);
        ITYPE basis_00 = insert_zero_to_basis_index( basis_insert_only_min , max_qubit_mask, max_qubit_index);
        ITYPE basis_01 = basis_00 ^ target_mask_0;
        ITYPE basis_10 = basis_00 ^ target_mask_1;
        swap_amplitude(state, basis_01, basis_10);
    }
}


void P0_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    ITYPE state_index;
    ITYPE mask = (1ULL << target_qubit_index);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0 ; state_index<loop_dim ; ++state_index){
        ITYPE temp_index = insert_zero_to_basis_index(state_index,mask,target_qubit_index) ^ mask;
        state[temp_index] = 0;
    }
}

void P1_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    ITYPE mask = (1ULL << target_qubit_index);
    ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0 ; state_index<loop_dim ; ++state_index){
        ITYPE temp_index = insert_zero_to_basis_index(state_index,mask,target_qubit_index);
        state[temp_index] = 0;
    }
}

void normalize(double norm, CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim;
    const double normalize_factor = sqrt(1./norm);
    ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0 ; state_index<loop_dim ; ++state_index){
        state[state_index] *= normalize_factor;
    }
}

void RX_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim){
    single_qubit_Pauli_rotation_gate(target_qubit_index, 1, angle, state, dim);
}

void RY_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim){
    single_qubit_Pauli_rotation_gate(target_qubit_index, 2, angle, state, dim);
}

void RZ_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim){
    single_qubit_Pauli_rotation_gate(target_qubit_index, 3, angle, state, dim);
}

void S_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_phase_gate(target_qubit_index, 1.i, state, dim);
}
void Sdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_phase_gate(target_qubit_index, -1.i, state, dim);
}
void T_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_phase_gate(target_qubit_index, (1.+1.i)/sqrt(2.), state, dim);
}
void Tdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_phase_gate(target_qubit_index, (1.-1.i)/sqrt(2.), state, dim);
}
void sqrtX_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_X_GATE_MATRIX, state, dim);
}
void sqrtXdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_X_DAG_GATE_MATRIX, state, dim);
}
void sqrtY_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_Y_GATE_MATRIX, state, dim);
}
void sqrtYdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_Y_DAG_GATE_MATRIX, state, dim);
}
