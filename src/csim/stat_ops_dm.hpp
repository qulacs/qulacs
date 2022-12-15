#pragma once

#include "type.hpp"

DllExport double dm_state_norm_squared(const CTYPE* state, ITYPE dim);
DllExport double dm_measurement_distribution_entropy(
    const CTYPE* state, ITYPE dim);
DllExport void dm_state_add(const CTYPE* state_added, CTYPE* state, ITYPE dim);
DllExport void dm_state_add_with_coef(
    CTYPE coef, const CTYPE* state_added, CTYPE* state, ITYPE dim);
DllExport void dm_state_multiply(CTYPE coef, CTYPE* state, ITYPE dim);

DllExport void dm_state_tensor_product(const CTYPE* state_left, ITYPE dim_left,
    const CTYPE* state_right, ITYPE dim_right, CTYPE* state_dst);
DllExport void dm_state_permutate_qubit(const UINT* qubit_order,
    const CTYPE* state_src, CTYPE* state_dst, UINT qubit_count, ITYPE dim);
DllExport void dm_state_partial_trace_from_density_matrix(const UINT* target,
    UINT target_count, const CTYPE* state_src, CTYPE* state_dst, ITYPE dim);
DllExport void dm_state_partial_trace_from_state_vector(const UINT* target,
    UINT target_count, const CTYPE* state_src, CTYPE* state_dst, ITYPE dim);

DllExport double dm_M0_prob(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim);
DllExport double dm_M1_prob(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim);
DllExport double dm_marginal_prob(const UINT* sorted_target_qubit_index_list,
    const UINT* measured_value_list, UINT target_qubit_index_count,
    const CTYPE* state, ITYPE dim);

DllExport double dm_expectation_value_multi_qubit_Pauli_operator_partial_list(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, const CTYPE* state, ITYPE dim);
