#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "stat_ops.h"
#include "utility.h"
#include "constant.h"

double expectation_value_X_Pauli_operator(UINT target_qubit_index, const CTYPE* state, ITYPE dim);
double expectation_value_Y_Pauli_operator(UINT target_qubit_index, const CTYPE* state, ITYPE dim);
double expectation_value_Z_Pauli_operator(UINT target_qubit_index, const CTYPE* state, ITYPE dim);
double expectation_value_multi_qubit_Pauli_operator_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,UINT pivot_qubit_index, const CTYPE* state, ITYPE dim);
double expectation_value_multi_qubit_Pauli_operator_Z_mask(ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim);

// calculate expectation value of X on target qubit
double expectation_value_X_Pauli_operator(UINT target_qubit_index, const CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum =0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(state_index=0;state_index<loop_dim;++state_index){
        ITYPE basis_0 = insert_zero_to_basis_index(state_index,mask,target_qubit_index);
        ITYPE basis_1 = basis_0 ^ mask;
        sum += creal( conj(state[basis_0]) * state[basis_1] ) * 2;
    }
    return sum;
}

// calculate expectation value of Y on target qubit
double expectation_value_Y_Pauli_operator(UINT target_qubit_index, const CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum =0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(state_index=0;state_index<loop_dim;++state_index){
        ITYPE basis_0 = insert_zero_to_basis_index(state_index,mask,target_qubit_index);
        ITYPE basis_1 = basis_0 ^ mask;
        sum += cimag( conj(state[basis_0]) * state[basis_1] ) * 2;
    }
    return sum;
}

// calculate expectation value of Z on target qubit
double expectation_value_Z_Pauli_operator(UINT target_qubit_index, const CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum =0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(state_index=0;state_index<loop_dim;++state_index){
        int sign = 1 - 2 * ((state_index >> target_qubit_index)%2);
        sum += creal( conj(state[state_index]) * state[state_index] ) * sign;
    }
    return sum;
}

// calculate expectation value for single-qubit pauli operator
double expectation_value_single_qubit_Pauli_operator(UINT target_qubit_index, UINT Pauli_operator_type, const CTYPE *state, ITYPE dim) {
    if(Pauli_operator_type == 0){
        return state_norm_squared(state,dim);
    }else if(Pauli_operator_type == 1){
        return expectation_value_X_Pauli_operator(target_qubit_index, state, dim);
    }else if(Pauli_operator_type == 2){
        return expectation_value_Y_Pauli_operator(target_qubit_index, state, dim);
    }else if(Pauli_operator_type == 3){
        return expectation_value_Z_Pauli_operator(target_qubit_index, state, dim);
    }else{
        fprintf(stderr,"invalid expectation value of pauli operator is called");
        exit(1);
    }
}


// calculate expectation value of multi-qubit Pauli operator on qubits.
// bit-flip mask : the n-bit binary string of which the i-th element is 1 iff the i-th pauli operator is X or Y
// phase-flip mask : the n-bit binary string of which the i-th element is 1 iff the i-th pauli operator is Y or Z
// We assume bit-flip mask is nonzero, namely, there is at least one X or Y operator.
// the pivot qubit is any qubit index which has X or Y
// To generate bit-flip mask and phase-flip mask, see get_masks_*_list at utility.h
double expectation_value_multi_qubit_Pauli_operator_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,UINT pivot_qubit_index, const CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    const ITYPE pivot_mask = 1ULL << pivot_qubit_index;
    ITYPE state_index;
    double sum = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(state_index=0;state_index<loop_dim;++state_index){
        ITYPE basis_0 = insert_zero_to_basis_index(state_index, pivot_mask, pivot_qubit_index);
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;
        UINT sign_0 = count_population(basis_0 & phase_flip_mask)%2;
        
        sum += creal(state[basis_0] * conj(state[basis_1]) * PHASE_90ROT[ (global_phase_90rot_count + sign_0*2)%4 ] * 2.0);
    }
    return sum;
}


double expectation_value_multi_qubit_Pauli_operator_Z_mask(ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(state_index=0;state_index<loop_dim;++state_index){
        int bit_parity = count_population(state_index & phase_flip_mask)%2;
        int sign = 1 - 2*bit_parity;
        sum += pow(cabs(state[state_index]),2) * sign;
    }
    return sum;
}

double expectation_value_multi_qubit_Pauli_operator_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, const CTYPE* state, ITYPE dim){
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    double result;
    if(bit_flip_mask == 0){
        result = expectation_value_multi_qubit_Pauli_operator_Z_mask(phase_flip_mask, state,dim);
    }else{
        result = expectation_value_multi_qubit_Pauli_operator_XZ_mask(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim);
    }
    return result;
}

double expectation_value_multi_qubit_Pauli_operator_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, const CTYPE* state, ITYPE dim){
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    double result;
    if(bit_flip_mask == 0){
        result = expectation_value_multi_qubit_Pauli_operator_Z_mask(phase_flip_mask, state, dim);
    }else{
        result = expectation_value_multi_qubit_Pauli_operator_XZ_mask(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim);
    }
    return result;
}

/****
 * Single thread version of expectation value
**/
// calculate expectation value of multi-qubit Pauli operator on qubits.
// bit-flip mask : the n-bit binary string of which the i-th element is 1 iff the i-th pauli operator is X or Y
// phase-flip mask : the n-bit binary string of which the i-th element is 1 iff the i-th pauli operator is Y or Z
// We assume bit-flip mask is nonzero, namely, there is at least one X or Y operator.
// the pivot qubit is any qubit index which has X or Y
// To generate bit-flip mask and phase-flip mask, see get_masks_*_list at utility.h
double expectation_value_multi_qubit_Pauli_operator_XZ_mask_single_thread(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,UINT pivot_qubit_index, const CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    const ITYPE pivot_mask = 1ULL << pivot_qubit_index;
    ITYPE state_index;
    double sum = 0.;
    UINT sign_0;
    ITYPE basis_0, basis_1;
    for(state_index=0;state_index<loop_dim;++state_index){
        basis_0 = insert_zero_to_basis_index(state_index, pivot_mask, pivot_qubit_index);
        basis_1 = basis_0 ^ bit_flip_mask;
        sign_0 = count_population(basis_0 & phase_flip_mask)%2;
        sum += creal(state[basis_0] * conj(state[basis_1]) * PHASE_90ROT[ (global_phase_90rot_count + sign_0*2)%4 ] * 2.0);
    }
    return sum;
}


double expectation_value_multi_qubit_Pauli_operator_Z_mask_single_thread(ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum = 0.;
    int bit_parity, sign;
    for(state_index=0;state_index<loop_dim;++state_index){
        bit_parity = count_population(state_index & phase_flip_mask)%2;
        sign = 1 - 2*bit_parity;
        sum += pow(cabs(state[state_index]),2) * sign;
    }
    return sum;
}

double expectation_value_multi_qubit_Pauli_operator_partial_list_single_thread(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, const CTYPE* state, ITYPE dim){
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    double result;
    if(bit_flip_mask == 0){
        result = expectation_value_multi_qubit_Pauli_operator_Z_mask_single_thread(phase_flip_mask, state,dim);
    }else{
        result = expectation_value_multi_qubit_Pauli_operator_XZ_mask_single_thread(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim);
    }
    return result;
}

