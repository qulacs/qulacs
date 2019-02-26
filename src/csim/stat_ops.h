#pragma once

#include "type.h"

DllExport double state_norm(const CTYPE *state, ITYPE dim) ;
DllExport double measurement_distribution_entropy(const CTYPE *state, ITYPE dim);
DllExport CTYPE state_inner_product(const CTYPE *state_bra, const CTYPE *state_ket, ITYPE dim);
DllExport void state_add(const CTYPE *state_added, CTYPE *state, ITYPE dim);
DllExport void state_multiply(CTYPE coef, CTYPE *state, ITYPE dim);

DllExport double M0_prob(UINT target_qubit_index, const CTYPE* state, ITYPE dim);
DllExport double M1_prob(UINT target_qubit_index, const CTYPE* state, ITYPE dim);
DllExport double marginal_prob(const UINT* sorted_target_qubit_index_list, const UINT* measured_value_list, UINT target_qubit_index_count, const CTYPE* state, ITYPE dim);

DllExport double expectation_value_single_qubit_Pauli_operator(UINT target_qubit_index, UINT Pauli_operator_type, const CTYPE *state, ITYPE dim);
DllExport double expectation_value_multi_qubit_Pauli_operator_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, const CTYPE* state, ITYPE dim);
DllExport double expectation_value_multi_qubit_Pauli_operator_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, const CTYPE* state, ITYPE dim);

DllExport CTYPE transition_amplitude_multi_qubit_Pauli_operator_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim);
DllExport CTYPE transition_amplitude_multi_qubit_Pauli_operator_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim);
