#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.hpp"
#include "stat_ops.hpp"
#include "utility.hpp"

// calculate probability with which we obtain 0 at target qubit
double M0_prob(UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum = 0.;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : sum)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_0 =
            insert_zero_to_basis_index(state_index, mask, target_qubit_index);
        sum += pow(_cabs(state[basis_0]), 2);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}

// calculate probability with which we obtain 1 at target qubit
double M1_prob(UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum = 0.;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : sum)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_1 =
            insert_zero_to_basis_index(state_index, mask, target_qubit_index) ^
            mask;
        sum += pow(_cabs(state[basis_1]), 2);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}

// calculate merginal probability with which we obtain the set of values
// measured_value_list at sorted_target_qubit_index_list warning:
// sorted_target_qubit_index_list must be sorted.
double marginal_prob(const UINT* sorted_target_qubit_index_list,
    const UINT* measured_value_list, UINT target_qubit_index_count,
    const CTYPE* state, ITYPE dim) {
    ITYPE loop_dim = dim >> target_qubit_index_count;
    ITYPE state_index;
    double sum = 0.;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : sum)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis = state_index;
        for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
            UINT insert_index = sorted_target_qubit_index_list[cursor];
            ITYPE mask = 1ULL << insert_index;
            basis = insert_zero_to_basis_index(basis, mask, insert_index);
            basis ^= mask * measured_value_list[cursor];
        }
        sum += pow(_cabs(state[basis]), 2);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}

// calculate entropy of probability distribution of Z-basis measurements
double measurement_distribution_entropy(const CTYPE* state, ITYPE dim) {
    ITYPE index;
    double ent = 0;
    const double eps = 1e-15;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : ent)
#endif
    for (index = 0; index < dim; ++index) {
        double prob = pow(_cabs(state[index]), 2);
        prob = (prob > eps) ? prob : eps;
        ent += -1.0 * prob * log(prob);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return ent;
}
