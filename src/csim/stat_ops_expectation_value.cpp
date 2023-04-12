#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "MPIutil.hpp"
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
double expectation_value_multi_qubit_Pauli_operator_XZ_mask_mpi(
    ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, const CTYPE* state, ITYPE dim, UINT inner_qc);
double expectation_value_multi_qubit_Pauli_operator_Z_mask_mpi(
    ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim, UINT rank,
    UINT inner_qc);

#ifdef _USE_SVE
double expectation_value_X_Pauli_operator_sve(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim);
double expectation_value_Y_Pauli_operator_sve(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim);
double expectation_value_Z_Pauli_operator_sve(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim);
double expectation_value_multi_qubit_Pauli_operator_XZ_mask_sve(
    ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, const CTYPE* state, ITYPE dim);
double expectation_value_multi_qubit_Pauli_operator_Z_mask_sve(
    ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim);
#endif

// calculate expectation value of X on target qubit
double expectation_value_X_Pauli_operator(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum = 0.;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#endif

#pragma omp parallel for reduction(+ : sum)
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_0 =
            insert_zero_to_basis_index(state_index, mask, target_qubit_index);
        ITYPE basis_1 = basis_0 ^ mask;
        sum += _creal(conj(state[basis_0]) * state[basis_1]) * 2;
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}

#ifdef _USE_SVE
double expectation_value_X_Pauli_operator_sve(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum = 0.;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#endif

    // # of complex128 numbers in an SVE registers
    ITYPE VL = svcntd() / 2;
#pragma omp parallel reduction(+ : sum)
    {
        svbool_t pall = svptrue_b64();

        svuint64_t sv_offset = svindex_u64(0, 1);
        svuint64_t sv_img_ofs = svindex_u64(0, 1);
        sv_offset = svlsr_z(pall, sv_offset, 1);
        sv_img_ofs = svand_x(pall, sv_img_ofs, svdup_u64(1));

        svfloat64_t sv_sum = svdup_f64(0.0);

#pragma omp for
        for (state_index = 0; state_index < loop_dim; state_index += VL) {
            svuint64_t sv_basis0 =
                svadd_z(pall, sv_offset, svdup_u64(state_index));
            svuint64_t sv_basis_tmp =
                svlsl_z(pall, svlsr_z(pall, sv_basis0, target_qubit_index),
                    target_qubit_index + 1);
            sv_basis0 = svadd_z(pall, sv_basis_tmp,
                svand_z(pall, sv_basis0, svdup_u64(mask - 1)));

            svuint64_t sv_basis1 = sveor_z(pall, sv_basis0, svdup_u64(mask));

            sv_basis0 = svmad_x(pall, sv_basis0, svdup_u64(2), sv_img_ofs);
            sv_basis1 = svmad_x(pall, sv_basis1, svdup_u64(2), sv_img_ofs);
            svfloat64_t sv_input0 =
                svld1_gather_index(pall, (double*)state, sv_basis0);
            svfloat64_t sv_input1 =
                svld1_gather_index(pall, (double*)state, sv_basis1);

            svfloat64_t sv_result = svmul_x(pall, sv_input0, sv_input1);
            sv_result = svmul_x(pall, sv_result, svdup_f64(2.0));
            sv_sum = svadd_x(pall, sv_sum, sv_result);
        }
        // reduction
        if (VL >= 16) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 16));
        if (VL >= 8) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 8));
        if (VL >= 4) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 4));
        if (VL >= 2) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 2));
        if (VL >= 1) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 1));

        sum += svlastb(svptrue_pat_b64(SV_VL1), sv_sum);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}
#endif

// calculate expectation value of Y on target qubit
double expectation_value_Y_Pauli_operator(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum = 0.;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#endif

#pragma omp parallel for reduction(+ : sum)
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_0 =
            insert_zero_to_basis_index(state_index, mask, target_qubit_index);
        ITYPE basis_1 = basis_0 ^ mask;
        sum += _cimag(conj(state[basis_0]) * state[basis_1]) * 2;
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}

#ifdef _USE_SVE
double expectation_value_Y_Pauli_operator_sve(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = 1ULL << target_qubit_index;
    ITYPE state_index;
    double sum = 0.;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#endif

    // # of complex128 numbers in an SVE registers
    ITYPE VL = svcntd() / 2;
#pragma omp parallel reduction(+ : sum)
    {
        svbool_t pall = svptrue_b64();
        svbool_t pg_conj_neg;

        svuint64_t sv_offset = svindex_u64(0, 1);
        svuint64_t sv_img_ofs = svindex_u64(0, 1);
        sv_offset = svlsr_z(pall, sv_offset, 1);
        sv_img_ofs = svand_x(pall, sv_img_ofs, svdup_u64(1));

        pg_conj_neg = svcmpeq(pall, sv_img_ofs, svdup_u64(1));

        svfloat64_t sv_sum = svdup_f64(0.0);

#pragma omp for
        for (state_index = 0; state_index < loop_dim; state_index += VL) {
            svuint64_t sv_basis0 =
                svadd_z(pall, sv_offset, svdup_u64(state_index));
            svuint64_t sv_basis_tmp =
                svlsl_z(pall, svlsr_z(pall, sv_basis0, target_qubit_index),
                    target_qubit_index + 1);
            sv_basis0 = svadd_z(pall, sv_basis_tmp,
                svand_z(pall, sv_basis0, svdup_u64(mask - 1)));

            svuint64_t sv_basis1 = sveor_z(pall, sv_basis0, svdup_u64(mask));

            sv_basis0 = svmad_x(pall, sv_basis0, svdup_u64(2), sv_img_ofs);
            sv_basis1 = svmad_x(pall, sv_basis1, svdup_u64(2), sv_img_ofs);
            svfloat64_t sv_input0 =
                svld1_gather_index(pall, (double*)state, sv_basis0);
            svfloat64_t sv_input1 =
                svld1_gather_index(pall, (double*)state, sv_basis1);

            svfloat64_t sv_real = svtrn1(sv_input0, sv_input1);
            svfloat64_t sv_imag = svtrn2(sv_input1, sv_input0);
            sv_imag = svneg_m(sv_imag, pg_conj_neg, sv_imag);
            svfloat64_t sv_result = svmul_x(pall, sv_real, sv_imag);
            sv_result = svmul_z(pall, sv_result, svdup_f64(2.0));
            sv_sum = svadd_z(pall, sv_sum, sv_result);
        }
        // reduction
        if (VL >= 16) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 16));
        if (VL >= 8) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 8));
        if (VL >= 4) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 4));
        if (VL >= 2) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 2));
        if (VL >= 1) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 1));

        sum += svlastb(svptrue_pat_b64(SV_VL1), sv_sum);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}
#endif

// calculate expectation value of Z on target qubit
double expectation_value_Z_Pauli_operator(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum = 0.;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#endif

#pragma omp parallel for reduction(+ : sum)
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        int sign = 1 - 2 * ((state_index >> target_qubit_index) % 2);
        sum += _creal(conj(state[state_index]) * state[state_index]) * sign;
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}

#ifdef _USE_SVE
double expectation_value_Z_Pauli_operator_sve(
    UINT target_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum = 0.;
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#endif

    // # of complex128 numbers in an SVE registers
    ITYPE VL = svcntd() / 2;
#pragma omp parallel reduction(+ : sum)
    {
        svbool_t pall = svptrue_b64();
        svuint64_t sv_offset = svindex_u64(0, 1);
        sv_offset = svlsr_z(pall, sv_offset, 1);

        svfloat64_t sv_sum = svdup_f64(0.0);

#pragma omp for
        for (state_index = 0; state_index < loop_dim; state_index += VL) {
            svuint64_t sv_sign =
                svadd_z(pall, svdup_u64(state_index), sv_offset);
            sv_sign = svlsr_z(pall, sv_sign, target_qubit_index);
            sv_sign = svand_z(pall, sv_sign, svdup_u64(1));
            svbool_t psign = svcmpeq(pall, sv_sign, svdup_u64(1));

            svfloat64_t sv_val = svld1(pall, (double*)&state[state_index]);
            sv_val = svmul_z(pall, sv_val, sv_val);
            sv_val = svneg_m(sv_val, psign, sv_val);
            sv_sum = svadd_z(pall, sv_sum, sv_val);
        }
        if (VL >= 16) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 16));
        if (VL >= 8) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 8));
        if (VL >= 4) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 4));
        if (VL >= 2) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 2));
        if (VL >= 1) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 1));

        sum += svlastb(svptrue_pat_b64(SV_VL1), sv_sum);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return sum;
}
#endif

// calculate expectation value for single-qubit pauli operator
double expectation_value_single_qubit_Pauli_operator(UINT target_qubit_index,
    UINT Pauli_operator_type, const CTYPE* state, ITYPE dim) {
#ifdef _USE_SVE
    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;
#endif

    if (Pauli_operator_type == 0) {
        return state_norm_squared(state, dim);
    } else if (Pauli_operator_type == 1) {
#ifdef _USE_SVE
        if (dim > VL) {
            return expectation_value_X_Pauli_operator_sve(
                target_qubit_index, state, dim);
        } else
#endif
        {
            return expectation_value_X_Pauli_operator(
                target_qubit_index, state, dim);
        }
    } else if (Pauli_operator_type == 2) {
#ifdef _USE_SVE
        if (dim > VL) {
            return expectation_value_Y_Pauli_operator_sve(
                target_qubit_index, state, dim);
        } else
#endif
        {
            return expectation_value_Y_Pauli_operator(
                target_qubit_index, state, dim);
        }
    } else if (Pauli_operator_type == 3) {
#ifdef _USE_SVE
        if (dim >= VL) {
            return expectation_value_Z_Pauli_operator_sve(
                target_qubit_index, state, dim);
        } else
#endif
        {
            return expectation_value_Z_Pauli_operator(
                target_qubit_index, state, dim);
        }
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
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_0 = insert_zero_to_basis_index(
            state_index, pivot_mask, pivot_qubit_index);
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;
        UINT sign_0 = count_population(basis_0 & phase_flip_mask) % 2;

        sum += _creal(state[basis_0] * conj(state[basis_1]) *
                      PHASE_90ROT[(global_phase_90rot_count + sign_0 * 2) % 4] *
                      2.0);
    }
    return sum;
}

#ifdef _USE_SVE
double expectation_value_multi_qubit_Pauli_operator_XZ_mask_sve(
    ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE pivot_mask = 1ULL << pivot_qubit_index;
    ITYPE state_index;
    double sum = 0.;

    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;

#ifdef _OPENMP
#pragma omp parallel reduction(+ : sum)
#endif
    {
        int img_flag = global_phase_90rot_count & 1;

        svbool_t pall = svptrue_b64();
        svbool_t pg_conj_neg;
        svuint64_t sv_idx_ofs = svindex_u64(0, 1);
        svuint64_t sv_img_ofs = sv_idx_ofs;

        sv_idx_ofs = svlsr_x(pall, sv_idx_ofs, 1);
        sv_img_ofs = svand_x(pall, sv_img_ofs, svdup_u64(1));
        pg_conj_neg = svcmpeq(pall, sv_img_ofs, svdup_u64(0));

        svfloat64_t sv_sum = svdup_f64(0.0);
        svfloat64_t sv_sign_base;
        if (global_phase_90rot_count & 2)
            sv_sign_base = svdup_f64(-1.0);
        else
            sv_sign_base = svdup_f64(1.0);

#pragma omp for
        for (state_index = 0; state_index < loop_dim; state_index += VL) {
            svuint64_t sv_basis =
                svadd_x(pall, svdup_u64(state_index), sv_idx_ofs);
            svuint64_t sv_basis0 = svlsr_x(pall, sv_basis, pivot_qubit_index);
            sv_basis0 = svlsl_x(pall, sv_basis0, pivot_qubit_index + 1);
            sv_basis0 = svadd_x(pall, sv_basis0,
                svand_x(pall, sv_basis, svdup_u64(pivot_mask - 1)));

            svuint64_t sv_basis1 =
                sveor_x(pall, sv_basis0, svdup_u64(bit_flip_mask));

            svuint64_t sv_popc =
                svand_x(pall, sv_basis0, svdup_u64(phase_flip_mask));
            sv_popc = svcnt_z(pall, sv_popc);
            sv_popc = svand_x(pall, sv_popc, svdup_u64(1));
            svfloat64_t sv_sign = svneg_m(sv_sign_base,
                svcmpeq(pall, sv_popc, svdup_u64(1)), sv_sign_base);
            sv_sign = svmul_x(pall, sv_sign, svdup_f64(2.0));

            sv_basis0 = svmad_x(pall, sv_basis0, svdup_u64(2), sv_img_ofs);
            sv_basis1 = svmad_x(pall, sv_basis1, svdup_u64(2), sv_img_ofs);
            svfloat64_t sv_input0 =
                svld1_gather_index(pall, (double*)state, sv_basis0);
            svfloat64_t sv_input1 =
                svld1_gather_index(pall, (double*)state, sv_basis1);

            if (img_flag) {  // calc imag. parts

                svfloat64_t sv_real = svtrn1(sv_input0, sv_input1);
                svfloat64_t sv_imag = svtrn2(sv_input1, sv_input0);
                sv_imag = svneg_m(sv_imag, pg_conj_neg, sv_imag);

                svfloat64_t sv_result = svmul_x(pall, sv_real, sv_imag);
                sv_result = svmul_x(pall, sv_result, sv_sign);
                sv_sum = svsub_x(pall, sv_sum, sv_result);

            } else {  // calc real parts

                svfloat64_t sv_result = svmul_x(pall, sv_input0, sv_input1);
                sv_result = svmul_x(pall, sv_result, sv_sign);
                sv_sum = svadd_x(pall, sv_sum, sv_result);
            }
        }

        // reduction
        if (VL >= 16) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 16));
        if (VL >= 8) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 8));
        if (VL >= 4) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 4));
        if (VL >= 2) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 2));
        if (VL >= 1) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 1));

        sum += svlastb(svptrue_pat_b64(SV_VL1), sv_sum);
    }
    return sum;
}
#endif

double expectation_value_multi_qubit_Pauli_operator_Z_mask(
    ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        int bit_parity = count_population(state_index & phase_flip_mask) % 2;
        int sign = 1 - 2 * bit_parity;
        sum += pow(_cabs(state[state_index]), 2) * sign;
    }
    return sum;
}

#ifdef _USE_SVE
double expectation_value_multi_qubit_Pauli_operator_Z_mask_sve(
    ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum = 0.;

    // # of complex128 numbers in an SVE registers
    ITYPE VL = svcntd() / 2;
#ifdef _OPENMP
#pragma omp parallel reduction(+ : sum)
#endif
    {
        svbool_t pall = svptrue_b64();
        svfloat64_t sv_sum = svdup_f64(0.0);
        svuint64_t sv_offset = svindex_u64(0, 1);
        svuint64_t sv_phase_flip_mask = svdup_u64(phase_flip_mask);

#pragma omp for
        for (state_index = 0; state_index < loop_dim;
             state_index += (VL << 1)) {
            svuint64_t svidx = svadd_z(pall, svdup_u64(state_index), sv_offset);
            svuint64_t sv_bit_parity = svand_z(pall, svidx, sv_phase_flip_mask);
            sv_bit_parity = svcnt_z(pall, sv_bit_parity);
            sv_bit_parity = svand_z(pall, sv_bit_parity, svdup_u64(1));

            svbool_t psign = svcmpeq(pall, sv_bit_parity, svdup_u64(1));

            svfloat64_t sv_val0 = svld1(pall, (double*)&state[state_index]);
            svfloat64_t sv_val1 =
                svld1(pall, (double*)&state[state_index + VL]);

            sv_val0 = svmul_z(pall, sv_val0, sv_val0);
            sv_val1 = svmul_z(pall, sv_val1, sv_val1);

            sv_val0 = svadd_z(pall, sv_val0, svext(sv_val0, sv_val0, 1));
            sv_val1 = svadd_z(pall, sv_val1, svext(sv_val1, sv_val1, 1));

            sv_val0 = svuzp1(sv_val0, sv_val1);
            sv_val0 = svneg_m(sv_val0, psign, sv_val0);

            sv_sum = svadd_z(pall, sv_sum, sv_val0);
        }

        if (VL >= 16) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 16));
        if (VL >= 8) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 8));
        if (VL >= 4) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 4));
        if (VL >= 2) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 2));
        if (VL >= 1) sv_sum = svadd_z(pall, sv_sum, svext(sv_sum, sv_sum, 1));

        sum += svlastb(svptrue_pat_b64(SV_VL1), sv_sum);
    }
    return sum;
}
#endif

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

#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#endif
#ifdef _USE_SVE
    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;
#endif

    if (bit_flip_mask == 0) {
#ifdef _USE_SVE
        if (dim > VL) {
            result = expectation_value_multi_qubit_Pauli_operator_Z_mask_sve(
                phase_flip_mask, state, dim);
        } else
#endif
        {
            result = expectation_value_multi_qubit_Pauli_operator_Z_mask(
                phase_flip_mask, state, dim);
        }
    } else {
#ifdef _USE_SVE
        if (dim > VL) {
            result = expectation_value_multi_qubit_Pauli_operator_XZ_mask_sve(
                bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
                pivot_qubit_index, state, dim);
        } else
#endif
        {
            result = expectation_value_multi_qubit_Pauli_operator_XZ_mask(
                bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
                pivot_qubit_index, state, dim);
        }
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
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

#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#endif
#ifdef _USE_SVE
    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;
#endif

    if (bit_flip_mask == 0) {
#ifdef _USE_SVE
        if (dim > VL) {
            result = expectation_value_multi_qubit_Pauli_operator_Z_mask_sve(
                phase_flip_mask, state, dim);
        } else
#endif
        {
            result = expectation_value_multi_qubit_Pauli_operator_Z_mask(
                phase_flip_mask, state, dim);
        }
    } else {
#ifdef _USE_SVE
        if (dim > VL) {
            result = expectation_value_multi_qubit_Pauli_operator_XZ_mask_sve(
                bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
                pivot_qubit_index, state, dim);
        } else
#endif
        {
            result = expectation_value_multi_qubit_Pauli_operator_XZ_mask(
                bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
                pivot_qubit_index, state, dim);
        }
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
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
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(1, 1);  // set num_thread=1
#endif

#ifdef _USE_SVE
    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;
#endif

    if (bit_flip_mask == 0) {
#ifdef _USE_SVE
        if (dim > VL) {
            result = expectation_value_multi_qubit_Pauli_operator_Z_mask_sve(
                phase_flip_mask, state, dim);
        } else
#endif
        {
            result = expectation_value_multi_qubit_Pauli_operator_Z_mask(
                phase_flip_mask, state, dim);
        }
    } else {
#ifdef _USE_SVE
        if (dim > VL) {
            result = expectation_value_multi_qubit_Pauli_operator_XZ_mask_sve(
                bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
                pivot_qubit_index, state, dim);
        } else
#endif
        {
            result = expectation_value_multi_qubit_Pauli_operator_XZ_mask(
                bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
                pivot_qubit_index, state, dim);
        }
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return result;
}

#ifdef _USE_MPI
double expectation_value_multi_qubit_Pauli_operator_partial_list_mpi(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, const CTYPE* state, ITYPE dim, UINT outer_qc,
    UINT inner_qc) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    double result;
    MPIutil& m = MPIutil::get_inst();

    if (bit_flip_mask == 0) {
        result = expectation_value_multi_qubit_Pauli_operator_Z_mask_mpi(
            phase_flip_mask, state, dim, m.get_rank(), inner_qc);
    } else {
        result = expectation_value_multi_qubit_Pauli_operator_XZ_mask_mpi(
            bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
            pivot_qubit_index, state, dim, inner_qc);
    }

    if (outer_qc > 0) MPIutil::get_inst().s_D_allreduce(&result);
    return result;
}

double expectation_value_multi_qubit_Pauli_operator_XZ_mask_mpi(
    ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, const CTYPE* state, ITYPE dim, UINT inner_qc) {
    const ITYPE pivot_mask = 1ULL << pivot_qubit_index;
    ITYPE state_index;
    double sum = 0.;

    int comm_flag = bit_flip_mask >> inner_qc;
    assert(comm_flag != 0);

    MPIutil& m = MPIutil::get_inst();
    int mpirank = m.get_rank();
    int pair_rank = mpirank ^ comm_flag;
    ITYPE global_offset = mpirank << inner_qc;

    if (comm_flag) {
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* recvptr = m.get_workarea(&dim_work, &num_work);
        ITYPE inner_mask = dim - 1;
        ITYPE i, j;

        state_index = 0;

        for (i = 0; i < num_work; ++i) {
            if (mpirank > pair_rank) {
                // sender
                const CTYPE* sendptr = state + dim_work * i;
                m.m_DC_send((void*)sendptr, dim_work, pair_rank);
            } else {
                // receiver
                m.m_DC_recv(recvptr, dim_work, pair_rank);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
                for (j = 0; j < dim_work; ++j) {
                    ITYPE basis_1 = state_index + j + (pair_rank << inner_qc);
                    ITYPE basis_0 = basis_1 ^ bit_flip_mask;
                    UINT sign_0 =
                        count_population(basis_0 & phase_flip_mask) % 2;

                    sum += _creal(
                        state[basis_0 & inner_mask] *
                        conj(recvptr[basis_1 & inner_mask]) *
                        PHASE_90ROT[(global_phase_90rot_count + sign_0 * 2) %
                                    4] *
                        2.0);
                }
                state_index += dim_work;
            }
        }
    } else {
        const ITYPE loop_dim = dim / 2;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            // A
            ITYPE basis_0 = insert_zero_to_basis_index(
                state_index, pivot_mask, pivot_qubit_index);
            // B
            ITYPE basis_1 = basis_0 ^ bit_flip_mask;
            // C
            UINT sign_0 =
                count_population((basis_0 + global_offset) & phase_flip_mask) %
                2;
            // D
            sum += _creal(
                state[basis_0] * conj(state[basis_1]) *
                PHASE_90ROT[(global_phase_90rot_count + sign_0 * 2) % 4] * 2.0);
        }
    }

    return sum;
}

double expectation_value_multi_qubit_Pauli_operator_Z_mask_mpi(
    ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim, UINT rank,
    UINT inner_qc) {
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum = 0.;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE global_index = state_index + (rank << inner_qc);
        // A
        int bit_parity = count_population(global_index & phase_flip_mask) % 2;
        // B
        double sign = 1 - 2 * bit_parity;
        // C
        sum += _creal(state[state_index] * conj(state[state_index])) * sign;
    }

    return sum;
}
#endif
