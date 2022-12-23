#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>

#include "constant.hpp"
#include "stat_ops.hpp"
#include "utility.hpp"

double expectation_value_X_Pauli_operator(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim);
double expectation_value_Y_Pauli_operator(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim);
double expectation_value_Z_Pauli_operator(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim);
double expectation_value_multi_qubit_Pauli_operator_XZ_mask(ITYPE bit_flip_mask,
    ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, const CTYPE* state, ITYPE dim);
double expectation_value_multi_qubit_Pauli_operator_Z_mask(
    ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim);

// calculate expectation value of X on target qubit
double expectation_value_X_Pauli_operator(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum = 0.;

#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : sum)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_0 =
            insert_zero_to_basis_index(state_index, mask, target_qubit_index);
        ITYPE basis_1 = basis_0 ^ mask;
        sum += _creal(conj(state[basis_0]) * state[basis_1]) * 2;
    }
#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
    return sum;
}

// calculate expectation value of Y on target qubit
double expectation_value_Y_Pauli_operator(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum = 0.;

#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : sum)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_0 =
            insert_zero_to_basis_index(state_index, mask, target_qubit_index);
        ITYPE basis_1 = basis_0 ^ mask;
        sum += _cimag(conj(state[basis_0]) * state[basis_1]) * 2;
    }
#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
    return sum;
}

// calculate expectation value of Z on target qubit
double expectation_value_Z_Pauli_operator(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum = 0.;

#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : sum)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        int sign = 1 - 2 * ((state_index >> target_qubit_index) % 2);
        sum += _creal(conj(state[state_index]) * state[state_index]) * sign;
    }
#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
    return sum;
}

// calculate expectation value for single-qubit pauli operator
double expectation_value_single_qubit_Pauli_operator(UINT target_qubit_index,
    UINT Pauli_operator_type, const CTYPE* state, ITYPE dim) {
    if (Pauli_operator_type == 0) {
        return state_norm_squared(state, dim);
    } else if (Pauli_operator_type == 1) {
        return expectation_value_X_Pauli_operator(
            target_qubit_index, state, dim);
    } else if (Pauli_operator_type == 2) {
        return expectation_value_Y_Pauli_operator(
            target_qubit_index, state, dim);
    } else if (Pauli_operator_type == 3) {
        return expectation_value_Z_Pauli_operator(
            target_qubit_index, state, dim);
    } else {
        fprintf(
            stderr, "invalid expectation value of pauli operator is called");
        exit(1);
    }
}

// calculate expectation value of multi-qubit Pauli operator on qubits.
// bit-flip mask : the n-bit binary string of which the i-th element is 1 iff
// the i-th pauli operator is X or Y phase-flip mask : the n-bit binary string
// of which the i-th element is 1 iff the i-th pauli operator is Y or Z We
// assume bit-flip mask is nonzero, namely, there is at least one X or Y
// operator. the pivot qubit is any qubit index which has X or Y To generate
// bit-flip mask and phase-flip mask, see get_masks_*_list at utility.h
double expectation_value_multi_qubit_Pauli_operator_XZ_mask(ITYPE bit_flip_mask,
    ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE pivot_mask = 1ULL << pivot_qubit_index;
    ITYPE state_index;
    double sum = 0.;

#ifdef _USE_SVE
    ITYPE vec_len = getVecLength();  // # of double elements in a vector

    if (loop_dim >= vec_len) {
#pragma omp parallel reduction(+ : sum)
        {
            int img_flag = global_phase_90rot_count & 1;

            SV_PRED pg = Svptrue();
            SV_PRED pg_conj_neg;
            SV_ITYPE sv_idx_ofs = SvindexI(0, 1);
            SV_ITYPE sv_img_ofs = SvindexI(0, 1);

            sv_idx_ofs = svlsr_x(pg, sv_idx_ofs, 1);
            sv_img_ofs = svand_x(pg, sv_img_ofs, SvdupI(1));
            pg_conj_neg = svcmpeq(pg, sv_img_ofs, SvdupI(0));

            SV_FTYPE sv_sum = SvdupF(0.0);
            SV_FTYPE sv_sign_base;
            if (global_phase_90rot_count & 2)
                sv_sign_base = SvdupF(-1.0);
            else
                sv_sign_base = SvdupF(1.0);

#pragma omp for
            for (state_index = 0; state_index < loop_dim;
                 state_index += (vec_len >> 1)) {
                SV_ITYPE sv_basis =
                    svadd_x(pg, SvdupI(state_index), sv_idx_ofs);
                SV_ITYPE sv_basis0 = svlsr_x(pg, sv_basis, pivot_qubit_index);
                sv_basis0 = svlsl_x(pg, sv_basis0, pivot_qubit_index + 1);
                sv_basis0 = svadd_x(pg, sv_basis0,
                    svand_x(pg, sv_basis, SvdupI(pivot_mask - 1)));

                SV_ITYPE sv_basis1 =
                    sveor_x(pg, sv_basis0, SvdupI(bit_flip_mask));

                SV_ITYPE sv_popc =
                    svand_x(pg, sv_basis0, SvdupI(phase_flip_mask));
                sv_popc = svcnt_z(pg, sv_popc);
                sv_popc = svand_x(pg, sv_popc, SvdupI(1));
                SV_FTYPE sv_sign = svneg_m(sv_sign_base,
                    svcmpeq(pg, sv_popc, SvdupI(1)), sv_sign_base);
                sv_sign = svmul_x(pg, sv_sign, SvdupF(2.0));

                sv_basis0 = svmad_x(pg, sv_basis0, SvdupI(2), sv_img_ofs);
                sv_basis1 = svmad_x(pg, sv_basis1, SvdupI(2), sv_img_ofs);
                SV_FTYPE sv_input0 =
                    svld1_gather_index(pg, (ETYPE*)state, sv_basis0);
                SV_FTYPE sv_input1 =
                    svld1_gather_index(pg, (ETYPE*)state, sv_basis1);

                if (img_flag) {  // calc imag. parts

                    SV_FTYPE sv_real = svtrn1(sv_input0, sv_input1);
                    SV_FTYPE sv_imag = svtrn2(sv_input1, sv_input0);
                    sv_imag = svneg_m(sv_imag, pg_conj_neg, sv_imag);

                    SV_FTYPE sv_result = svmul_x(pg, sv_real, sv_imag);
                    sv_result = svmul_x(pg, sv_result, sv_sign);
                    sv_sum = svsub_x(pg, sv_sum, sv_result);

                } else {  // calc real parts

                    SV_FTYPE sv_result = svmul_x(pg, sv_input0, sv_input1);
                    sv_result = svmul_x(pg, sv_result, sv_sign);
                    sv_sum = svadd_x(pg, sv_sum, sv_result);
                }
            }

            // reduction
            if (vec_len >= 32)
                sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 16));
            if (vec_len >= 16)
                sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 8));
            if (vec_len >= 8)
                sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 4));
            if (vec_len >= 4)
                sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 2));
            if (vec_len >= 2)
                sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 1));

            sum += svlastb(svptrue_pat_b64(SV_VL1), sv_sum);
        }
    } else
#endif
    {
#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : sum)
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_0 = insert_zero_to_basis_index(
                state_index, pivot_mask, pivot_qubit_index);
            ITYPE basis_1 = basis_0 ^ bit_flip_mask;
            UINT sign_0 = count_population(basis_0 & phase_flip_mask) % 2;

            sum += _creal(
                state[basis_0] * conj(state[basis_1]) *
                PHASE_90ROT[(global_phase_90rot_count + sign_0 * 2) % 4] * 2.0);
        }
    }
#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
    return sum;
}

double expectation_value_multi_qubit_Pauli_operator_Z_mask(
    ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum = 0.;

#ifdef _USE_SVE
    ITYPE vec_len = getVecLength();  // # of double elements in a vector
    if (loop_dim >= vec_len) {
#pragma omp parallel reduction(+ : sum)
        {
            SV_PRED pg = Svptrue();
            SV_FTYPE sv_sum = SvdupF(0.0);
            SV_ITYPE sv_offset = SvindexI(0, 1);
            SV_ITYPE sv_phase_flip_mask = SvdupI(phase_flip_mask);

#pragma omp for
            for (state_index = 0; state_index < loop_dim;
                 state_index += vec_len) {
                ITYPE global_index = state_index;

                SV_ITYPE svidx = svadd_z(pg, SvdupI(global_index), sv_offset);
                SV_ITYPE sv_bit_parity = svand_z(pg, svidx, sv_phase_flip_mask);
                sv_bit_parity = svcnt_z(pg, sv_bit_parity);
                sv_bit_parity = svand_z(pg, sv_bit_parity, SvdupI(1));

                SV_PRED pg_sign = svcmpeq(pg, sv_bit_parity, SvdupI(1));

                SV_FTYPE sv_val0 = svld1(pg, (ETYPE*)&state[state_index]);
                SV_FTYPE sv_val1 =
                    svld1(pg, (ETYPE*)&state[state_index + (vec_len >> 1)]);

                sv_val0 = svmul_z(pg, sv_val0, sv_val0);
                sv_val1 = svmul_z(pg, sv_val1, sv_val1);

                sv_val0 = svadd_z(pg, sv_val0, svext(sv_val0, sv_val0, 1));
                sv_val1 = svadd_z(pg, sv_val1, svext(sv_val1, sv_val1, 1));

                sv_val0 = svuzp1(sv_val0, sv_val1);
                sv_val0 = svneg_m(sv_val0, pg_sign, sv_val0);

                sv_sum = svadd_z(pg, sv_sum, sv_val0);
            }

            if (vec_len >= 32)
                sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 16));
            if (vec_len >= 16)
                sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 8));
            if (vec_len >= 8)
                sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 4));
            if (vec_len >= 4)
                sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 2));
            if (vec_len >= 2)
                sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 1));

            sum += svlastb(svptrue_pat_b64(SV_VL1), sv_sum);
        }
    } else
#endif
    {

#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : sum)
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            int bit_parity =
                count_population(state_index & phase_flip_mask) % 2;
            int sign = 1 - 2 * bit_parity;
            sum += pow(_cabs(state[state_index]), 2) * sign;
        }
    }
#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
    return sum;
}

double expectation_value_multi_qubit_Pauli_operator_partial_list(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, const CTYPE* state, ITYPE dim) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    double result;
    if (bit_flip_mask == 0) {
        result = expectation_value_multi_qubit_Pauli_operator_Z_mask(
            phase_flip_mask, state, dim);
    } else {
        result = expectation_value_multi_qubit_Pauli_operator_XZ_mask(
            bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
            pivot_qubit_index, state, dim);
    }
    return result;
}

double expectation_value_multi_qubit_Pauli_operator_whole_list(
    const UINT* Pauli_operator_type_list, UINT qubit_count, const CTYPE* state,
    ITYPE dim) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count,
        &pivot_qubit_index);
    double result;
    if (bit_flip_mask == 0) {
        result = expectation_value_multi_qubit_Pauli_operator_Z_mask(
            phase_flip_mask, state, dim);
    } else {
        result = expectation_value_multi_qubit_Pauli_operator_XZ_mask(
            bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
            pivot_qubit_index, state, dim);
    }
    return result;
}

/****
 * Single thread version of expectation value
 **/
// calculate expectation value of multi-qubit Pauli operator on qubits.
// bit-flip mask : the n-bit binary string of which the i-th element is 1 iff
// the i-th pauli operator is X or Y phase-flip mask : the n-bit binary string
// of which the i-th element is 1 iff the i-th pauli operator is Y or Z We
// assume bit-flip mask is nonzero, namely, there is at least one X or Y
// operator. the pivot qubit is any qubit index which has X or Y To generate
// bit-flip mask and phase-flip mask, see get_masks_*_list at utility.h
double expectation_value_multi_qubit_Pauli_operator_XZ_mask_single_thread(
    ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE pivot_mask = 1ULL << pivot_qubit_index;
    ITYPE state_index;
    double sum = 0.;
    UINT sign_0;
    ITYPE basis_0, basis_1;
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        basis_0 = insert_zero_to_basis_index(
            state_index, pivot_mask, pivot_qubit_index);
        basis_1 = basis_0 ^ bit_flip_mask;
        sign_0 = count_population(basis_0 & phase_flip_mask) % 2;
        sum += _creal(state[basis_0] * conj(state[basis_1]) *
                      PHASE_90ROT[(global_phase_90rot_count + sign_0 * 2) % 4] *
                      2.0);
    }
    return sum;
}

double expectation_value_multi_qubit_Pauli_operator_Z_mask_single_thread(
    ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum = 0.;
    int bit_parity, sign;
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        bit_parity = count_population(state_index & phase_flip_mask) % 2;
        sign = 1 - 2 * bit_parity;
        sum += pow(_cabs(state[state_index]), 2) * sign;
    }
    return sum;
}

double expectation_value_multi_qubit_Pauli_operator_partial_list_single_thread(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, const CTYPE* state, ITYPE dim) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    double result;
    if (bit_flip_mask == 0) {
        result =
            expectation_value_multi_qubit_Pauli_operator_Z_mask_single_thread(
                phase_flip_mask, state, dim);
    } else {
        result =
            expectation_value_multi_qubit_Pauli_operator_XZ_mask_single_thread(
                bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
                pivot_qubit_index, state, dim);
    }
    return result;
}
