#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.hpp"
#include "stat_ops.hpp"
#include "utility.hpp"

CTYPE
transition_amplitude_multi_qubit_Pauli_operator_XZ_mask(ITYPE bit_flip_mask,
    ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, const CTYPE* state_bra, const CTYPE* state_ket,
    ITYPE dim);
CTYPE
transition_amplitude_multi_qubit_Pauli_operator_Z_mask(ITYPE phase_flip_mask,
    const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim);

CTYPE
transition_amplitude_multi_qubit_Pauli_operator_XZ_mask(ITYPE bit_flip_mask,
    ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, const CTYPE* state_bra, const CTYPE* state_ket,
    ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE pivot_mask = 1ULL << pivot_qubit_index;
    ITYPE state_index;

    double sum_real = 0.;
    double sum_imag = 0.;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : sum_real, sum_imag)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_0 = insert_zero_to_basis_index(
            state_index, pivot_mask, pivot_qubit_index);
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;
        UINT sign_0 = count_population(basis_0 & phase_flip_mask) % 2;
        UINT sign_1 = count_population(basis_1 & phase_flip_mask) % 2;
        CTYPE val1 = state_ket[basis_0] * conj(state_bra[basis_1]) *
                     PHASE_90ROT[(global_phase_90rot_count + sign_0 * 2) % 4];
        CTYPE val2 = state_ket[basis_1] * conj(state_bra[basis_0]) *
                     PHASE_90ROT[(global_phase_90rot_count + sign_1 * 2) % 4];
        sum_real += _creal(val1);
        sum_imag += _cimag(val1);
        sum_real += _creal(val2);
        sum_imag += _cimag(val2);
    }
    CTYPE sum(sum_real, sum_imag);
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}

CTYPE
transition_amplitude_multi_qubit_Pauli_operator_Z_mask(ITYPE phase_flip_mask,
    const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim) {
    const ITYPE loop_dim = dim;
    ITYPE state_index;

    double sum_real = 0.;
    double sum_imag = 0.;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : sum_real, sum_imag)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        int bit_parity = count_population(state_index & phase_flip_mask) % 2;
        double sign = 1 - 2 * bit_parity;
        CTYPE val =
            sign * state_ket[state_index] * conj(state_bra[state_index]);
        sum_real += _creal(val);
        sum_imag += _cimag(val);
    }
    CTYPE sum(sum_real, sum_imag);
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}

CTYPE
transition_amplitude_multi_qubit_Pauli_operator_partial_list(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, const CTYPE* state_bra,
    const CTYPE* state_ket, ITYPE dim) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    CTYPE result;
    if (bit_flip_mask == 0) {
        result = transition_amplitude_multi_qubit_Pauli_operator_Z_mask(
            phase_flip_mask, state_bra, state_ket, dim);
    } else {
        result = transition_amplitude_multi_qubit_Pauli_operator_XZ_mask(
            bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
            pivot_qubit_index, state_bra, state_ket, dim);
    }
    return result;
}

CTYPE
transition_amplitude_multi_qubit_Pauli_operator_whole_list(
    const UINT* Pauli_operator_type_list, UINT qubit_count,
    const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count,
        &pivot_qubit_index);
    CTYPE result;
    if (bit_flip_mask == 0) {
        result = transition_amplitude_multi_qubit_Pauli_operator_Z_mask(
            phase_flip_mask, state_bra, state_ket, dim);
    } else {
        result = transition_amplitude_multi_qubit_Pauli_operator_XZ_mask(
            bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
            pivot_qubit_index, state_bra, state_ket, dim);
    }
    return result;
}
