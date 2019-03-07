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
CTYPE transition_amplitude_multi_qubit_Pauli_operator_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim);
CTYPE transition_amplitude_multi_qubit_Pauli_operator_Z_mask(ITYPE phase_flip_mask, const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim);

// calculate norm
double state_norm(const CTYPE *state, ITYPE dim) {
    ITYPE index;
    double norm = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:norm)
#endif
    for (index = 0; index < dim; ++index){
        norm += pow(cabs(state[index]), 2);
    }
    return norm;
}

// calculate entropy of probability distribution of Z-basis measurements
double measurement_distribution_entropy(const CTYPE *state, ITYPE dim){
    ITYPE index;
    double ent=0;
    const double eps = 1e-15;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:ent)
#endif
    for(index = 0; index < dim; ++index){
        double prob = pow(cabs(state[index]),2);
        if(prob > eps){
            ent += -1.0*prob*log(prob);
        } 
    }
    return ent;
}

// calculate inner product of two state vector
CTYPE state_inner_product(const CTYPE *state_bra, const CTYPE *state_ket, ITYPE dim) {
#ifndef _MSC_VER
    CTYPE value = 0;
    ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:value)
#endif
    for(index = 0; index < dim; ++index){
        value += conj(state_bra[index]) * state_ket[index];
    }
    return value;
#else

    double real_sum = 0.;
    double imag_sum = 0.;
    ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:real_sum,imag_sum)
#endif
    for (index = 0; index < dim; ++index) {
        CTYPE value;
        value += conj(state_bra[index]) * state_ket[index];
        real_sum += creal(value);
        imag_sum += cimag(value);
    }
    return real_sum + 1.i * imag_sum;
#endif
}


void state_add(const CTYPE *state_added, CTYPE *state, ITYPE dim) {
	ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (index = 0; index < dim; ++index) {
		state[index] += state_added[index];
	}
}

void state_multiply(CTYPE coef, CTYPE *state, ITYPE dim) {
	ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (index = 0; index < dim; ++index) {
		state[index] *= coef;
	}
}


// calculate probability with which we obtain 0 at target qubit
double M0_prob(UINT target_qubit_index, const CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum =0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(state_index=0;state_index<loop_dim;++state_index){
        ITYPE basis_0 = insert_zero_to_basis_index(state_index,mask,target_qubit_index);
        sum += pow(cabs(state[basis_0]),2);
    }
    return sum;
}

// calculate probability with which we obtain 1 at target qubit
double M1_prob(UINT target_qubit_index, const CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum =0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(state_index=0;state_index<loop_dim;++state_index){
        ITYPE basis_1 = insert_zero_to_basis_index(state_index,mask,target_qubit_index) ^ mask;
        sum += pow(cabs(state[basis_1]),2);
    }
    return sum;
}

// calculate merginal probability with which we obtain the set of values measured_value_list at sorted_target_qubit_index_list
// warning: sorted_target_qubit_index_list must be sorted.
double marginal_prob(const UINT* sorted_target_qubit_index_list, const UINT* measured_value_list, UINT target_qubit_index_count, const CTYPE* state, ITYPE dim){
    ITYPE loop_dim = dim >> target_qubit_index_count;
    ITYPE state_index;
    double sum=0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(state_index = 0;state_index < loop_dim; ++state_index){
        ITYPE basis = state_index;
        for(UINT cursor=0; cursor < target_qubit_index_count ; cursor++){
            UINT insert_index = sorted_target_qubit_index_list[cursor];
            ITYPE mask = 1ULL << insert_index;
            basis = insert_zero_to_basis_index(basis, mask , insert_index );
            basis ^= mask * measured_value_list[cursor];
        }
        sum += pow(cabs(state[basis]),2);
    }
    return sum;
}

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
        return state_norm(state,dim);
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

CTYPE transition_amplitude_multi_qubit_Pauli_operator_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE pivot_mask = 1ULL << pivot_qubit_index;
    ITYPE state_index;


#ifndef _MSC_VER
    CTYPE sum = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_0 = insert_zero_to_basis_index(state_index, pivot_mask, pivot_qubit_index);
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;
        UINT sign_0 = count_population(basis_0 & phase_flip_mask) % 2;
        sum += state_ket[basis_0] * conj(state_bra[basis_1]) * PHASE_90ROT[(global_phase_90rot_count + sign_0 * 2) % 4];

        UINT sign_1 = count_population(basis_1 & phase_flip_mask) % 2;
        sum += state_ket[basis_1] * conj(state_bra[basis_0]) * PHASE_90ROT[(global_phase_90rot_count + sign_1 * 2) % 4];
    }

#else
    double sum_real = 0.;
    double sum_imag = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum_real, sum_imag)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_0 = insert_zero_to_basis_index(state_index, pivot_mask, pivot_qubit_index);
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;
        UINT sign_0 = count_population(basis_0 & phase_flip_mask) % 2;
        UINT sign_1 = count_population(basis_1 & phase_flip_mask) % 2;
        CTYPE val1 = state_ket[basis_0] * conj(state_bra[basis_1]) * PHASE_90ROT[(global_phase_90rot_count + sign_0 * 2) % 4];
        CTYPE val2 = state_ket[basis_1] * conj(state_bra[basis_0]) * PHASE_90ROT[(global_phase_90rot_count + sign_1 * 2) % 4];
        sum_real += creal(val1);
        sum_imag += cimag(val1);
        sum_real += creal(val2);
        sum_imag += cimag(val2);
    }
    CTYPE sum(sum_real, sum_imag);
#endif
    return sum;
}


CTYPE transition_amplitude_multi_qubit_Pauli_operator_Z_mask(ITYPE phase_flip_mask, const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim) {
    const ITYPE loop_dim = dim;
    ITYPE state_index;

#ifndef _MSC_VER
    CTYPE sum = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        int bit_parity = count_population(state_index & phase_flip_mask) % 2;
        double sign = 1 - 2 * bit_parity;
        sum += sign*state_ket[state_index] * conj(state_bra[state_index]);
    }
    return sum;

#else

    double sum_real = 0.;
    double sum_imag = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum_real, sum_imag)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        int bit_parity = count_population(state_index & phase_flip_mask) % 2;
        double sign = 1 - 2 * bit_parity;
        CTYPE val = sign * state_ket[state_index] * conj(state_bra[state_index]);
        sum_real += creal(val);
        sum_imag += cimag(val);
    }
    CTYPE sum(sum_real, sum_imag);
#endif
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


CTYPE transition_amplitude_multi_qubit_Pauli_operator_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    CTYPE result;
    if (bit_flip_mask == 0) {
        result = transition_amplitude_multi_qubit_Pauli_operator_Z_mask(phase_flip_mask, state_bra, state_ket, dim);
    }
    else {
        result = transition_amplitude_multi_qubit_Pauli_operator_XZ_mask(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_bra, state_ket, dim);
    }
    return result;
}

CTYPE transition_amplitude_multi_qubit_Pauli_operator_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    CTYPE result;
    if (bit_flip_mask == 0) {
        result = transition_amplitude_multi_qubit_Pauli_operator_Z_mask(phase_flip_mask, state_bra, state_ket, dim);
    }
    else {
        result = transition_amplitude_multi_qubit_Pauli_operator_XZ_mask(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_bra, state_ket, dim);
    }
    return result;
}


