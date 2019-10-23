#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "stat_ops_dm.h"
#include "utility.h"
#include "constant.h"

// calculate norm
double dm_state_norm(const CTYPE *state, ITYPE dim) {
    ITYPE index;
    double norm = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:norm)
#endif
    for (index = 0; index < dim; ++index){
        norm += creal(state[index*dim+index]);
    }
    return norm;
}

// calculate entropy of probability distribution of Z-basis measurements
double dm_measurement_distribution_entropy(const CTYPE *state, ITYPE dim){
    ITYPE index;
    double ent=0;
    const double eps = 1e-15;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:ent)
#endif
    for(index = 0; index < dim; ++index){
		double prob = creal(state[index*dim + index]);
        if(prob > eps){
            ent += -1.0*prob*log(prob);
        } 
    }
    return ent;
}

// calculate probability with which we obtain 0 at target qubit
double dm_M0_prob(UINT target_qubit_index, const CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum =0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(state_index=0;state_index<loop_dim;++state_index){
        ITYPE basis_0 = insert_zero_to_basis_index(state_index,mask,target_qubit_index);
        sum += creal(state[basis_0*dim+basis_0]);
    }
    return sum;
}

// calculate probability with which we obtain 1 at target qubit
double dm_M1_prob(UINT target_qubit_index, const CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum =0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
    for(state_index=0;state_index<loop_dim;++state_index){
        ITYPE basis_1 = insert_zero_to_basis_index(state_index,mask,target_qubit_index) ^ mask;
		sum += creal(state[basis_1*dim + basis_1]);
	}
    return sum;
}

// calculate merginal probability with which we obtain the set of values measured_value_list at sorted_target_qubit_index_list
// warning: sorted_target_qubit_index_list must be sorted.
double dm_marginal_prob(const UINT* sorted_target_qubit_index_list, const UINT* measured_value_list, UINT target_qubit_index_count, const CTYPE* state, ITYPE dim){
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
        sum += creal(state[basis*dim+basis]);
    }
    return sum;
}


void dm_state_add(const CTYPE *state_added, CTYPE *state, ITYPE dim) {
	ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (index = 0; index < dim*dim; ++index) {
		state[index] += state_added[index];
	}
}

void dm_state_multiply(CTYPE coef, CTYPE *state, ITYPE dim) {
	ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (index = 0; index < dim*dim; ++index) {
		state[index] *= coef;
	}
}


double dm_expectation_value_multi_qubit_Pauli_operator_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, const CTYPE* state, ITYPE dim) {
	const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
	CTYPE* matrix = (CTYPE*)malloc(sizeof(CTYPE)*matrix_dim*matrix_dim);
	for (ITYPE y = 0; y < matrix_dim; ++y) {
		for (ITYPE x = 0; x < matrix_dim; ++x) {
			CTYPE coef = 1.0;
			for (UINT i = 0; i < target_qubit_index_count; ++i) {
				ITYPE xi = (x >> i) % 2;
				ITYPE yi = (y >> i) % 2;
				coef *= PAULI_MATRIX[Pauli_operator_type_list[i]][yi * 2 + xi];
			}
			matrix[y*matrix_dim + x] = coef;
		}
	}
	const ITYPE* matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);

	CTYPE sum = 0;
	for (ITYPE state_index = 0; state_index< dim; ++state_index) {
		ITYPE small_dim_index = 0;
		ITYPE basis_0 = state_index;
		for (UINT i = 0; i < target_qubit_index_count; ++i) {
			UINT target_qubit_index = target_qubit_index_list[i];
			if (state_index & (1ULL << target_qubit_index)) {
				small_dim_index += (1ULL << i);
				basis_0 ^= (1ULL << target_qubit_index);
			}
		}
		for (ITYPE i = 0; i < matrix_dim; ++i) {
			sum += matrix[small_dim_index*matrix_dim + i] * state[state_index*dim + (basis_0 ^ matrix_mask_list[i])];
		}
	}

	free(matrix);
	free((ITYPE*)matrix_mask_list);
	return creal(sum);
}
