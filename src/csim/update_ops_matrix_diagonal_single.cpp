
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

void single_qubit_diagonal_matrix_gate(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 12);
#endif

#ifdef _USE_SIMD
    single_qubit_diagonal_matrix_gate_parallel_simd(
        target_qubit_index, diagonal_matrix, state, dim);
#elif defined(_USE_SVE)
    single_qubit_diagonal_matrix_gate_parallel_sve(
        target_qubit_index, diagonal_matrix, state, dim);
#else
    single_qubit_diagonal_matrix_gate_parallel_unroll(
        target_qubit_index, diagonal_matrix, state, dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void single_qubit_diagonal_matrix_gate_parallel_unroll(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    if (target_qubit_index == 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            state[state_index] *= diagonal_matrix[0];
            state[state_index + 1] *= diagonal_matrix[1];
        }
    } else {
        ITYPE mask = 1ULL << target_qubit_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            int bitval = ((state_index & mask) != 0);
            state[state_index] *= diagonal_matrix[bitval];
            state[state_index + 1] *= diagonal_matrix[bitval];
        }
    }
}

#ifdef _USE_SIMD
void single_qubit_diagonal_matrix_gate_parallel_simd(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    if (target_qubit_index == 0) {
        __m256d mv0 = _mm256_set_pd(-_cimag(diagonal_matrix[1]),
            _creal(diagonal_matrix[1]), -_cimag(diagonal_matrix[0]),
            _creal(diagonal_matrix[0]));
        __m256d mv1 = _mm256_set_pd(_creal(diagonal_matrix[1]),
            _cimag(diagonal_matrix[1]), _creal(diagonal_matrix[0]),
            _cimag(diagonal_matrix[0]));
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            double *ptr = (double *)(state + state_index);
            __m256d data = _mm256_loadu_pd(ptr);
            __m256d data0 = _mm256_mul_pd(data, mv0);
            __m256d data1 = _mm256_mul_pd(data, mv1);
            data = _mm256_hadd_pd(data0, data1);
            _mm256_storeu_pd(ptr, data);
        }
    } else {
        __m256d mv0 = _mm256_set_pd(-_cimag(diagonal_matrix[0]),
            _creal(diagonal_matrix[0]), -_cimag(diagonal_matrix[0]),
            _creal(diagonal_matrix[0]));
        __m256d mv1 = _mm256_set_pd(_creal(diagonal_matrix[0]),
            _cimag(diagonal_matrix[0]), _creal(diagonal_matrix[0]),
            _cimag(diagonal_matrix[0]));
        __m256d mv2 = _mm256_set_pd(-_cimag(diagonal_matrix[1]),
            _creal(diagonal_matrix[1]), -_cimag(diagonal_matrix[1]),
            _creal(diagonal_matrix[1]));
        __m256d mv3 = _mm256_set_pd(_creal(diagonal_matrix[1]),
            _cimag(diagonal_matrix[1]), _creal(diagonal_matrix[1]),
            _cimag(diagonal_matrix[1]));
        //__m256i mask = _mm256_set1_epi64x(1LL<<target_qubit_index);
        ITYPE mask = 1LL << target_qubit_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            double *ptr = (double *)(state + state_index);
            ITYPE flag = (state_index & mask);
            __m256d mv4 = flag ? mv2 : mv0;
            __m256d mv5 = flag ? mv3 : mv1;
            __m256d data = _mm256_loadu_pd(ptr);
            __m256d data0 = _mm256_mul_pd(data, mv4);
            __m256d data1 = _mm256_mul_pd(data, mv5);
            data = _mm256_hadd_pd(data0, data1);
            _mm256_storeu_pd(ptr, data);
        }
    }
}
#endif

#ifdef _USE_SVE
void single_qubit_diagonal_matrix_gate_parallel_sve(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    ITYPE mask = 1ULL << target_qubit_index;

    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;

    if (mask > VL) {
#pragma omp parallel
        {
            svbool_t pg = svptrue_b64();

            svfloat64_t mat0r, mat0i, mat1r, mat1i;

            mat0r = svdup_f64(_creal(diagonal_matrix[0]));
            mat0i = svdup_f64(_cimag(diagonal_matrix[0]));
            mat1r = svdup_f64(_creal(diagonal_matrix[1]));
            mat1i = svdup_f64(_cimag(diagonal_matrix[1]));

#pragma omp for
            for (state_index = 0; state_index < loop_dim;
                 state_index += (VL << 1)) {
                int bitval = ((state_index & mask) != 0);

                // fetch values
                svfloat64_t input0 = svld1(pg, (double *)&state[state_index]);
                svfloat64_t input1 =
                    svld1(pg, (double *)&state[state_index + VL]);

                // select matrix elements
                svfloat64_t matr = (bitval != 0) ? mat1r : mat0r;
                svfloat64_t mati = (bitval != 0) ? mat1i : mat0i;

                // select odd or even elements from two vectors
                svfloat64_t cvalr = svuzp1(input0, input1);
                svfloat64_t cvali = svuzp2(input0, input1);

                // perform complex multiplication
                svfloat64_t resultr = svmul_x(pg, cvalr, matr);
                svfloat64_t resulti = svmul_x(pg, cvali, matr);

                resultr = svmsb_x(pg, cvali, mati, resultr);
                resulti = svmad_x(pg, cvalr, mati, resulti);

                // interleave elements from low halves of two vectors
                svfloat64_t output0 = svzip1(resultr, resulti);
                svfloat64_t output1 = svzip2(resultr, resulti);

                // set values
                svst1(pg, (double *)&state[state_index], output0);
                svst1(pg, (double *)&state[state_index + VL], output1);
            }
        }
    } else {
        if (loop_dim > VL) {
#pragma omp parallel
            {
                svbool_t pg = svptrue_b64();

                svfloat64_t mat0r, mat0i, mat1r, mat1i;

                // SVE registers for control factor elements
                svuint64_t vec_index_diff;
                vec_index_diff = svindex_u64(0, 1);  // (0, 1, 2, 3, 4,...)

                mat0r = svdup_f64(_creal(diagonal_matrix[0]));
                mat0i = svdup_f64(_cimag(diagonal_matrix[0]));
                mat1r = svdup_f64(_creal(diagonal_matrix[1]));
                mat1i = svdup_f64(_cimag(diagonal_matrix[1]));

#pragma omp for
                for (state_index = 0; state_index < loop_dim;
                     state_index += (VL << 1)) {
                    // fetch values
                    svfloat64_t input0 =
                        svld1(pg, (double *)&state[state_index]);
                    svfloat64_t input1 =
                        svld1(pg, (double *)&state[state_index + VL]);

                    // select matrix elements
                    svuint64_t vec_bitval =
                        svadd_z(pg, vec_index_diff, svdup_u64(state_index));
                    vec_bitval = svand_z(pg, vec_bitval, svdup_u64(mask));

                    svbool_t select_flag =
                        svcmpne(pg, vec_bitval, svdup_u64(0));

                    svfloat64_t matr = svsel(select_flag, mat1r, mat0r);
                    svfloat64_t mati = svsel(select_flag, mat1i, mat0i);

                    // select odd or even elements from two vectors
                    svfloat64_t cvalr = svuzp1(input0, input1);
                    svfloat64_t cvali = svuzp2(input0, input1);

                    // perform complex multiplication
                    svfloat64_t resultr = svmul_x(pg, cvalr, matr);
                    svfloat64_t resulti = svmul_x(pg, cvali, matr);

                    resultr = svmsb_x(pg, cvali, mati, resultr);
                    resulti = svmad_x(pg, cvalr, mati, resulti);

                    // interleave elements from low halves of two vectors
                    svfloat64_t output0 = svzip1(resultr, resulti);
                    svfloat64_t output1 = svzip2(resultr, resulti);

                    // set values
                    svst1(pg, (double *)&state[state_index], output0);
                    svst1(pg, (double *)&state[state_index + VL], output1);
                }
            }
        } else {
#pragma omp parallel for
            for (state_index = 0; state_index < loop_dim; state_index++) {
                int bitval = ((state_index & mask) != 0);
                state[state_index] *= diagonal_matrix[bitval];
            }
        }
    }
}
#endif

#ifdef _USE_MPI
void single_qubit_diagonal_matrix_gate_partial(
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim, int isone);

void single_qubit_diagonal_matrix_gate_mpi(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        single_qubit_diagonal_matrix_gate(
            target_qubit_index, diagonal_matrix, state, dim);
    } else {
#ifdef _OPENMP
        OMPutil::get_inst().set_qulacs_num_threads(dim, 12);
#endif
        const int rank = MPIutil::get_inst().get_rank();
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);

        single_qubit_diagonal_matrix_gate_partial(
            diagonal_matrix, state, dim, (rank & pair_rank_bit) != 0);
#ifdef _OPENMP
        OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    }
}

void single_qubit_diagonal_matrix_gate_partial(
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim, int isone) {
    // loop variables
    ITYPE state_index;

    {
#pragma omp parallel for
        for (state_index = 0; state_index < dim; state_index += 2) {
            state[state_index] *= diagonal_matrix[isone];
            state[state_index + 1] *= diagonal_matrix[isone];
        }
    }
}
#endif
