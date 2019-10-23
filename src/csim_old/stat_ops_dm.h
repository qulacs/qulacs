#pragma once

#include "type.h"

DllExport double dm_state_norm(const CTYPE *state, ITYPE dim) ;
DllExport double dm_measurement_distribution_entropy(const CTYPE *state, ITYPE dim);
DllExport void dm_state_add(const CTYPE *state_added, CTYPE *state, ITYPE dim);
DllExport void dm_state_multiply(CTYPE coef, CTYPE *state, ITYPE dim);

DllExport double dm_M0_prob(UINT target_qubit_index, const CTYPE* state, ITYPE dim);
DllExport double dm_M1_prob(UINT target_qubit_index, const CTYPE* state, ITYPE dim);
DllExport double dm_marginal_prob(const UINT* sorted_target_qubit_index_list, const UINT* measured_value_list, UINT target_qubit_index_count, const CTYPE* state, ITYPE dim);

DllExport double dm_expectation_value_multi_qubit_Pauli_operator_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, const CTYPE* state, ITYPE dim);
