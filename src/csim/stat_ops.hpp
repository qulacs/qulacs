#pragma once

#include "type.hpp"

DllExport double state_norm_squared(const CTYPE* state, ITYPE dim);
DllExport double state_norm_squared_single_thread(
    const CTYPE* state, ITYPE dim);
DllExport double state_norm_squared_mpi(const CTYPE* state, ITYPE dim);

DllExport double measurement_distribution_entropy(
    const CTYPE* state, ITYPE dim);
DllExport CTYPE state_inner_product(
    const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim);
DllExport CTYPE state_inner_product_mpi(const CTYPE* state_bra,
    const CTYPE* state_ket, ITYPE dim_bra, ITYPE dim_ket);

DllExport void state_tensor_product(const CTYPE* state_left, ITYPE dim_left,
    const CTYPE* state_right, ITYPE dim_right, CTYPE* state_dst);
DllExport void state_permutate_qubit(const UINT* qubit_order,
    const CTYPE* state_src, CTYPE* state_dst, UINT qubit_count, ITYPE dim);
DllExport void state_drop_qubits(const UINT* target, const UINT* projection,
    UINT target_count, const CTYPE* state_src, CTYPE* state_dst, ITYPE dim);

DllExport double M0_prob(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim);
DllExport double M1_prob(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim);
DllExport double marginal_prob(const UINT* sorted_target_qubit_index_list,
    const UINT* measured_value_list, UINT target_qubit_index_count,
    const CTYPE* state, ITYPE dim);

DllExport double expectation_value_single_qubit_Pauli_operator(
    UINT target_qubit_index, UINT Pauli_operator_type, const CTYPE* state,
    ITYPE dim);
DllExport double expectation_value_multi_qubit_Pauli_operator_whole_list(
    const UINT* Pauli_operator_type_list, UINT qubit_count, const CTYPE* state,
    ITYPE dim);
DllExport double expectation_value_multi_qubit_Pauli_operator_partial_list(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, const CTYPE* state, ITYPE dim);

DllExport CTYPE transition_amplitude_multi_qubit_Pauli_operator_whole_list(
    const UINT* Pauli_operator_type_list, UINT qubit_count,
    const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim);
DllExport CTYPE transition_amplitude_multi_qubit_Pauli_operator_partial_list(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, const CTYPE* state_bra,
    const CTYPE* state_ket, ITYPE dim);

DllExport double
expectation_value_multi_qubit_Pauli_operator_partial_list_single_thread(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, const CTYPE* state, ITYPE dim);
DllExport double expectation_value_multi_qubit_Pauli_operator_partial_list_mpi(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, const CTYPE* state, ITYPE dim, UINT outer_qc,
    UINT inner_qc);
