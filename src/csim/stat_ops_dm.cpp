#include "stat_ops_dm.hpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.hpp"
#include "utility.hpp"

// calculate norm
double dm_state_norm_squared(const CTYPE* state, ITYPE dim) {
    ITYPE index;
    double norm = 0;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : norm)
#endif
    for (index = 0; index < dim; ++index) {
        norm += _creal(state[index * dim + index]);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return norm;
}

// calculate entropy of probability distribution of Z-basis measurements
double dm_measurement_distribution_entropy(const CTYPE* state, ITYPE dim) {
    ITYPE index;
    double ent = 0;
    const double eps = 1e-15;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : ent)
#endif
    for (index = 0; index < dim; ++index) {
        double prob = _creal(state[index * dim + index]);
        if (prob > eps) {
            ent += -1.0 * prob * log(prob);
        }
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return ent;
}

// calculate probability with which we obtain 0 at target qubit
double dm_M0_prob(UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
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
        sum += _creal(state[basis_0 * dim + basis_0]);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}

// calculate probability with which we obtain 1 at target qubit
double dm_M1_prob(UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
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
        sum += _creal(state[basis_1 * dim + basis_1]);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}

// calculate merginal probability with which we obtain the set of values
// measured_value_list at sorted_target_qubit_index_list warning:
// sorted_target_qubit_index_list must be sorted.
double dm_marginal_prob(const UINT* sorted_target_qubit_index_list,
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
        sum += _creal(state[basis * dim + basis]);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}

void dm_state_add(const CTYPE* state_added, CTYPE* state, ITYPE dim) {
    ITYPE index;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#pragma omp parallel for
#endif
    for (index = 0; index < dim * dim; ++index) {
        state[index] += state_added[index];
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void dm_state_add_with_coef(
    CTYPE coef, const CTYPE* state_added, CTYPE* state, ITYPE dim) {
    ITYPE index;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#pragma omp parallel for
#endif
    for (index = 0; index < dim * dim; ++index) {
        state[index] += coef * state_added[index];
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void dm_state_multiply(CTYPE coef, CTYPE* state, ITYPE dim) {
    ITYPE index;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#pragma omp parallel for
#endif
    for (index = 0; index < dim * dim; ++index) {
        state[index] *= coef;
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

double dm_expectation_value_multi_qubit_Pauli_operator_partial_list(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, const CTYPE* state, ITYPE dim) {
    CTYPE sum = 0;
    for (ITYPE state_index = 0; state_index < dim; ++state_index) {
        CTYPE coef = 1.0;
        ITYPE state_index_sub = state_index;
        for (UINT i = 0; i < target_qubit_index_count; ++i) {
            UINT pauli_type = Pauli_operator_type_list[i];
            UINT target_qubit_index = target_qubit_index_list[i];
            if (pauli_type == 1) {
                state_index_sub ^= ((1ULL) << target_qubit_index);
            } else if (pauli_type == 2) {
                coef *= 1.i;
                if (state_index_sub & ((1ULL) << target_qubit_index)) {
                    coef *= -1.;
                }
                state_index_sub ^= ((1ULL) << target_qubit_index);
            } else if (pauli_type == 3) {
                if (state_index_sub & ((1ULL) << target_qubit_index)) {
                    coef *= -1.;
                }
            }
        }
        sum += coef * state[state_index * dim + state_index_sub];
    }
    return _creal(sum);
}

void dm_state_tensor_product(const CTYPE* state_left, ITYPE dim_left,
    const CTYPE* state_right, ITYPE dim_right, CTYPE* state_dst) {
    ITYPE y_left, x_left, y_right, x_right;
    const ITYPE dim_new = dim_left * dim_right;
    for (y_left = 0; y_left < dim_left; ++y_left) {
        for (x_left = 0; x_left < dim_left; ++x_left) {
            CTYPE val_left = state_left[y_left * dim_left + x_left];
            for (y_right = 0; y_right < dim_right; ++y_right) {
                for (x_right = 0; x_right < dim_right; ++x_right) {
                    CTYPE val_right =
                        state_right[y_right * dim_right + x_right];
                    ITYPE x_new = x_left * dim_right + x_right;
                    ITYPE y_new = y_left * dim_right + y_right;
                    state_dst[y_new * dim_new + x_new] = val_right * val_left;
                }
            }
        }
    }
}

void dm_state_permutate_qubit(const UINT* qubit_order, const CTYPE* state_src,
    CTYPE* state_dst, UINT qubit_count, ITYPE dim) {
    ITYPE y, x;
    for (y = 0; y < dim; ++y) {
        for (x = 0; x < dim; ++x) {
            ITYPE src_x = 0, src_y = 0;
            for (UINT qubit_index = 0; qubit_index < qubit_count;
                 ++qubit_index) {
                if ((x >> qubit_index) % 2) {
                    src_x += 1ULL << qubit_order[qubit_index];
                }
                if ((y >> qubit_index) % 2) {
                    src_y += 1ULL << qubit_order[qubit_index];
                }
            }
            state_dst[y * dim + x] = state_src[src_y * dim + src_x];
        }
    }
}

void dm_state_partial_trace_from_density_matrix(const UINT* target,
    UINT target_count, const CTYPE* state_src, CTYPE* state_dst, ITYPE dim) {
    ITYPE dst_dim = dim >> target_count;
    ITYPE trace_dim = 1ULL << target_count;
    UINT* sorted_target = create_sorted_ui_list(target, target_count);
    ITYPE* mask_list = create_matrix_mask_list(target, target_count);

    ITYPE y, x;
    for (y = 0; y < dst_dim; ++y) {
        for (x = 0; x < dst_dim; ++x) {
            ITYPE base_x = x;
            ITYPE base_y = y;
            for (UINT target_index = 0; target_index < target_count;
                 ++target_index) {
                UINT insert_index = sorted_target[target_index];
                base_x = insert_zero_to_basis_index(
                    base_x, 1ULL << insert_index, insert_index);
                base_y = insert_zero_to_basis_index(
                    base_y, 1ULL << insert_index, insert_index);
            }
            CTYPE val = 0.;
            for (ITYPE idx = 0; idx < trace_dim; ++idx) {
                ITYPE src_x = base_x ^ mask_list[idx];
                ITYPE src_y = base_y ^ mask_list[idx];
                val += state_src[src_y * dim + src_x];
            }
            state_dst[y * dst_dim + x] = val;
        }
    }
    free(sorted_target);
    free(mask_list);
}

void dm_state_partial_trace_from_state_vector(const UINT* target,
    UINT target_count, const CTYPE* state_src, CTYPE* state_dst, ITYPE dim) {
    ITYPE dst_dim = dim >> target_count;
    ITYPE trace_dim = 1ULL << target_count;
    UINT* sorted_target = create_sorted_ui_list(target, target_count);
    ITYPE* mask_list = create_matrix_mask_list(target, target_count);

    ITYPE y, x;
    for (y = 0; y < dst_dim; ++y) {
        for (x = 0; x < dst_dim; ++x) {
            ITYPE base_x = x;
            ITYPE base_y = y;
            for (UINT target_index = 0; target_index < target_count;
                 ++target_index) {
                UINT insert_index = sorted_target[target_index];
                base_x = insert_zero_to_basis_index(
                    base_x, 1ULL << insert_index, insert_index);
                base_y = insert_zero_to_basis_index(
                    base_y, 1ULL << insert_index, insert_index);
            }
            CTYPE val = 0.;
            for (ITYPE idx = 0; idx < trace_dim; ++idx) {
                ITYPE src_x = base_x ^ mask_list[idx];
                ITYPE src_y = base_y ^ mask_list[idx];
                val += state_src[src_y] * conj(state_src[src_x]);
            }
            state_dst[y * dst_dim + x] = val;
        }
    }
    free(sorted_target);
    free(mask_list);
}
