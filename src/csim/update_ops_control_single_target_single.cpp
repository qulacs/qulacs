
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "MPIutil.hpp"
#include "constant.hpp"
#include "update_ops.hpp"
#include "utility.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void single_qubit_control_single_qubit_dense_matrix_gate(
    UINT control_qubit_index, UINT control_value, UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    single_qubit_control_single_qubit_dense_matrix_gate_simd(
        control_qubit_index, control_value, target_qubit_index, matrix, state,
        dim);
#elif defined(_USE_SVE)
    single_qubit_control_single_qubit_dense_matrix_gate_sve512(
        control_qubit_index, control_value, target_qubit_index, matrix, state,
        dim);
#else
    single_qubit_control_single_qubit_dense_matrix_gate_unroll(
        control_qubit_index, control_value, target_qubit_index, matrix, state,
        dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void single_qubit_control_single_qubit_dense_matrix_gate_unroll(
    UINT control_qubit_index, UINT control_value, UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;

    const UINT min_qubit_index =
        get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index =
        get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index;
    if (target_qubit_index == 0) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;

            // fetch values
            CTYPE cval0 = state[basis_index];
            CTYPE cval1 = state[basis_index + 1];

            // set values
            state[basis_index] = matrix[0] * cval0 + matrix[1] * cval1;
            state[basis_index + 1] = matrix[2] * cval0 + matrix[3] * cval1;
        }
    } else if (control_qubit_index == 0) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_index_1 = basis_index_0 + target_mask;

            // fetch values
            CTYPE cval0 = state[basis_index_0];
            CTYPE cval1 = state[basis_index_1];

            // set values
            state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
            state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
        }
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_index_1 = basis_index_0 + target_mask;

            // fetch values
            CTYPE cval0 = state[basis_index_0];
            CTYPE cval1 = state[basis_index_1];
            CTYPE cval2 = state[basis_index_0 + 1];
            CTYPE cval3 = state[basis_index_1 + 1];

            // set values
            state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
            state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
            state[basis_index_0 + 1] = matrix[0] * cval2 + matrix[1] * cval3;
            state[basis_index_1 + 1] = matrix[2] * cval2 + matrix[3] * cval3;
        }
    }
}

#ifdef _USE_SIMD
void single_qubit_control_single_qubit_dense_matrix_gate_simd(
    UINT control_qubit_index, UINT control_value, UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;

    const UINT min_qubit_index =
        get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index =
        get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index;
    if (target_qubit_index == 0) {
        __m256d mv00 = _mm256_set_pd(-_cimag(matrix[1]), _creal(matrix[1]),
            -_cimag(matrix[0]), _creal(matrix[0]));
        __m256d mv01 = _mm256_set_pd(_creal(matrix[1]), _cimag(matrix[1]),
            _creal(matrix[0]), _cimag(matrix[0]));
        __m256d mv20 = _mm256_set_pd(-_cimag(matrix[3]), _creal(matrix[3]),
            -_cimag(matrix[2]), _creal(matrix[2]));
        __m256d mv21 = _mm256_set_pd(_creal(matrix[3]), _cimag(matrix[3]),
            _creal(matrix[2]), _cimag(matrix[2]));
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            double* ptr = (double*)(state + basis);
            __m256d data = _mm256_loadu_pd(ptr);

            __m256d data_u0 = _mm256_mul_pd(data, mv00);
            __m256d data_u1 = _mm256_mul_pd(data, mv01);
            __m256d data_u2 = _mm256_hadd_pd(data_u0, data_u1);
            data_u2 = _mm256_permute4x64_pd(data_u2,
                216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

            __m256d data_d0 = _mm256_mul_pd(data, mv20);
            __m256d data_d1 = _mm256_mul_pd(data, mv21);
            __m256d data_d2 = _mm256_hadd_pd(data_d0, data_d1);
            data_d2 = _mm256_permute4x64_pd(data_d2,
                216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

            __m256d data_r = _mm256_hadd_pd(data_u2, data_d2);

            data_r = _mm256_permute4x64_pd(data_r,
                216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216
            _mm256_storeu_pd(ptr, data_r);
        }
    } else if (control_qubit_index == 0) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_index_1 = basis_index_0 + target_mask;

            // fetch values
            CTYPE cval0 = state[basis_index_0];
            CTYPE cval1 = state[basis_index_1];

            // set values
            state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
            state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
        }
    } else {
        __m256d mv00 = _mm256_set_pd(-_cimag(matrix[0]), _creal(matrix[0]),
            -_cimag(matrix[0]), _creal(matrix[0]));
        __m256d mv01 = _mm256_set_pd(_creal(matrix[0]), _cimag(matrix[0]),
            _creal(matrix[0]), _cimag(matrix[0]));
        __m256d mv10 = _mm256_set_pd(-_cimag(matrix[1]), _creal(matrix[1]),
            -_cimag(matrix[1]), _creal(matrix[1]));
        __m256d mv11 = _mm256_set_pd(_creal(matrix[1]), _cimag(matrix[1]),
            _creal(matrix[1]), _cimag(matrix[1]));
        __m256d mv20 = _mm256_set_pd(-_cimag(matrix[2]), _creal(matrix[2]),
            -_cimag(matrix[2]), _creal(matrix[2]));
        __m256d mv21 = _mm256_set_pd(_creal(matrix[2]), _cimag(matrix[2]),
            _creal(matrix[2]), _cimag(matrix[2]));
        __m256d mv30 = _mm256_set_pd(-_cimag(matrix[3]), _creal(matrix[3]),
            -_cimag(matrix[3]), _creal(matrix[3]));
        __m256d mv31 = _mm256_set_pd(_creal(matrix[3]), _cimag(matrix[3]),
            _creal(matrix[3]), _cimag(matrix[3]));
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_1 = basis_0 + target_mask;

            double* ptr0 = (double*)(state + basis_0);
            double* ptr1 = (double*)(state + basis_1);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            __m256d data1 = _mm256_loadu_pd(ptr1);

            __m256d data_u2 = _mm256_mul_pd(data0, mv00);
            __m256d data_u3 = _mm256_mul_pd(data1, mv10);
            __m256d data_u4 = _mm256_mul_pd(data0, mv01);
            __m256d data_u5 = _mm256_mul_pd(data1, mv11);

            __m256d data_u6 = _mm256_hadd_pd(data_u2, data_u4);
            __m256d data_u7 = _mm256_hadd_pd(data_u3, data_u5);

            __m256d data_d2 = _mm256_mul_pd(data0, mv20);
            __m256d data_d3 = _mm256_mul_pd(data1, mv30);
            __m256d data_d4 = _mm256_mul_pd(data0, mv21);
            __m256d data_d5 = _mm256_mul_pd(data1, mv31);

            __m256d data_d6 = _mm256_hadd_pd(data_d2, data_d4);
            __m256d data_d7 = _mm256_hadd_pd(data_d3, data_d5);

            __m256d data_r0 = _mm256_add_pd(data_u6, data_u7);
            __m256d data_r1 = _mm256_add_pd(data_d6, data_d7);

            _mm256_storeu_pd(ptr0, data_r0);
            _mm256_storeu_pd(ptr1, data_r1);
        }
    }
}
#endif

#ifdef _USE_SVE

static inline void MatrixVectorProduct2x2(svbool_t pg, svfloat64_t in00r,
    svfloat64_t in00i, svfloat64_t in11r, svfloat64_t in11i, svfloat64_t mat02r,
    svfloat64_t mat02i, svfloat64_t mat13r, svfloat64_t mat13i,
    svfloat64_t* out01r, svfloat64_t* out01i) {
    *out01r = svmul_x(pg, in00r, mat02r);
    *out01i = svmul_x(pg, in00i, mat02r);

    *out01r = svmsb_x(pg, in00i, mat02i, *out01r);
    *out01i = svmad_x(pg, in00r, mat02i, *out01i);

    *out01r = svmad_x(pg, in11r, mat13r, *out01r);
    *out01i = svmad_x(pg, in11r, mat13i, *out01i);

    *out01r = svmsb_x(pg, in11i, mat13i, *out01r);
    *out01i = svmad_x(pg, in11i, mat13r, *out01i);
}

void single_qubit_control_single_qubit_dense_matrix_gate_sve512(
    UINT control_qubit_index, UINT control_value, UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;

    const UINT min_qubit_index =
        get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index =
        get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index = 0;

    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;

    if ((VL == 4) && (dim > VL) && (min_qubit_mask >= VL)) {
#pragma omp parallel
        {
            // Create an all 1's predicate variable
            svbool_t pg = svptrue_b64();

            svfloat64_t mat02r, mat02i, mat13r, mat13i;

            // Load matrix elements to SVE variables
            // e.g.) mat02_real is [matrix[0].real, matrix[0].real,
            // matrix[2].real, matrix[2].real],
            //       if # of elements in SVE variables is four.
            mat02r = svuzp1(
                svdup_f64(_creal(matrix[0])), svdup_f64(_creal(matrix[2])));
            mat02i = svuzp1(
                svdup_f64(_cimag(matrix[0])), svdup_f64(_cimag(matrix[2])));
            mat13r = svuzp1(
                svdup_f64(_creal(matrix[1])), svdup_f64(_creal(matrix[3])));
            mat13i = svuzp1(
                svdup_f64(_cimag(matrix[1])), svdup_f64(_cimag(matrix[3])));

#pragma omp for
            for (state_index = 0; state_index < loop_dim; state_index += VL) {
                // Calculate indices
                ITYPE basis_0 = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) +
                                control_mask * control_value;
                ITYPE basis_1 = basis_0 + target_mask;

                // Load values
                svfloat64_t input0 = svld1(pg, (double*)&state[basis_0]);
                svfloat64_t input1 = svld1(pg, (double*)&state[basis_1]);

                // Select odd or even elements from two vectors
                svfloat64_t cval00r = svuzp1(input0, input0);
                svfloat64_t cval00i = svuzp2(input0, input0);
                svfloat64_t cval11r = svuzp1(input1, input1);
                svfloat64_t cval11i = svuzp2(input1, input1);

                // Perform matrix-vector products
                svfloat64_t result01r, result01i;
                MatrixVectorProduct2x2(pg, cval00r, cval00i, cval11r, cval11i,
                    mat02r, mat02i, mat13r, mat13i, &result01r, &result01i);
                // Interleave elements from low or high halves of two
                // vectors
                svfloat64_t output0 = svzip1(result01r, result01i);
                svfloat64_t output1 = svzip2(result01r, result01i);

                // Store values
                svst1(pg, (double*)&state[basis_0], output0);
                svst1(pg, (double*)&state[basis_1], output1);
            }
        }
    } else {
        single_qubit_control_single_qubit_dense_matrix_gate_unroll(
            control_qubit_index, control_value, target_qubit_index, matrix,
            state, dim);
    }
}
#endif

#ifdef _USE_MPI
void single_qubit_control_single_qubit_dense_matrix_gate_mpi_OI(
    UINT control_qubit_index, UINT control_value, CTYPE* t,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim, int flag,
    UINT index_offset);
void single_qubit_control_single_qubit_dense_matrix_gate_mpi_OO(
    CTYPE* t, const CTYPE matrix[4], CTYPE* state, ITYPE dim, int flag);

void single_qubit_control_single_qubit_dense_matrix_gate_mpi(
    UINT control_qubit_index, UINT control_value, UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim, UINT inner_qc) {
    MPIutil& m = MPIutil::get_inst();
    const UINT rank = m.get_rank();
    ITYPE dim_work = dim;
    ITYPE num_work = 0;
    CTYPE* t = m.get_workarea(&dim_work, &num_work);
    assert(num_work > 0);
    CTYPE* si = state;

    if (target_qubit_index < inner_qc) {
        if (control_qubit_index < inner_qc) {  // control, target: inner, inner
            single_qubit_control_single_qubit_dense_matrix_gate(
                control_qubit_index, control_value, target_qubit_index, matrix,
                state, dim);
        } else {  // control, target: outer, inner
            const UINT control_rank_bit = 1 << (control_qubit_index - inner_qc);
            if (((rank & control_rank_bit) && (control_value == 1)) ||
                (!(rank & control_rank_bit) && (control_value == 0)))
                single_qubit_dense_matrix_gate(
                    target_qubit_index, matrix, state, dim);
        }
    } else {
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        const int pair_rank = rank ^ pair_rank_bit;
#ifdef _OPENMP
        OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif
        if (control_qubit_index < inner_qc) {  // control, target: inner, outer
            for (ITYPE iter = 0; iter < num_work; ++iter) {
                m.m_DC_sendrecv(si, t, dim_work, pair_rank);

                UINT index_offset = iter * dim_work;
                single_qubit_control_single_qubit_dense_matrix_gate_mpi_OI(
                    control_qubit_index, control_value, t, matrix, si, dim_work,
                    rank & pair_rank_bit, index_offset);

                si += dim_work;
            }
        } else {  // control, target: outer, outer
            const UINT control_rank_bit = 1 << (control_qubit_index - inner_qc);
            ITYPE dummy_flag =
                !(((rank & control_rank_bit) && (control_value == 1)) ||
                    (!(rank & control_rank_bit) && (control_value == 0)));
            for (ITYPE iter = 0; iter < num_work; ++iter) {
                if (dummy_flag) {  // only count up tag
                    m.get_tag();
                } else {
                    m.m_DC_sendrecv(si, t, dim_work, pair_rank);

                    single_qubit_control_single_qubit_dense_matrix_gate_mpi_OO(
                        t, matrix, si, dim_work, rank & pair_rank_bit);

                    si += dim_work;
                }
            }
        }
#ifdef _OPENMP
        OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    }
}

void single_qubit_control_single_qubit_dense_matrix_gate_mpi_OI(
    UINT control_qubit_index, UINT control_value, CTYPE* t,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim, int flag,
    UINT index_offset) {
    UINT control_qubit_mask = 1ULL << control_qubit_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (ITYPE state_index = 0; state_index < dim; ++state_index) {
        UINT skip_flag = (state_index + index_offset) & control_qubit_mask;
        skip_flag = skip_flag >> control_qubit_index;
        if (skip_flag != control_value) continue;

        if (flag) {  // val=1
            // fetch values
            CTYPE cval_0 = t[state_index];
            CTYPE cval_1 = state[state_index];

            // set values
            state[state_index] = matrix[2] * cval_0 + matrix[3] * cval_1;
        } else {  // val=0
            // fetch values
            CTYPE cval_0 = state[state_index];
            CTYPE cval_1 = t[state_index];

            // set values
            state[state_index] = matrix[0] * cval_0 + matrix[1] * cval_1;
        }
    }
}

void single_qubit_control_single_qubit_dense_matrix_gate_mpi_OO(
    CTYPE* t, const CTYPE matrix[4], CTYPE* state, ITYPE dim, int flag) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (ITYPE state_index = 0; state_index < dim; ++state_index) {
        if (flag) {  // val=1
            // fetch values
            CTYPE cval_0 = t[state_index];
            CTYPE cval_1 = state[state_index];

            // set values
            state[state_index] = matrix[2] * cval_0 + matrix[3] * cval_1;
        } else {  // val=0
            // fetch values
            CTYPE cval_0 = state[state_index];
            CTYPE cval_1 = t[state_index];

            // set values
            state[state_index] = matrix[0] * cval_0 + matrix[1] * cval_1;
        }
    }
}
#endif
