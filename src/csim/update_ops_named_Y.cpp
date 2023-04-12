
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

void Y_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    Y_gate_parallel_simd(target_qubit_index, state, dim);
#elif defined(_USE_SVE)
    Y_gate_parallel_sve(target_qubit_index, state, dim);
#else
    Y_gate_parallel_unroll(target_qubit_index, state, dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void Y_gate_parallel_unroll(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    const CTYPE imag = 1.i;
    if (target_qubit_index == 0) {
        ITYPE basis_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (basis_index = 0; basis_index < dim; basis_index += 2) {
            CTYPE temp0 = state[basis_index];
            state[basis_index] = -imag * state[basis_index + 1];
            state[basis_index + 1] = imag * temp0;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_index_1 = basis_index_0 + mask;
            CTYPE temp0 = state[basis_index_0];
            CTYPE temp1 = state[basis_index_0 + 1];
            state[basis_index_0] = -imag * state[basis_index_1];
            state[basis_index_0 + 1] = -imag * state[basis_index_1 + 1];
            state[basis_index_1] = imag * temp0;
            state[basis_index_1 + 1] = imag * temp1;
        }
    }
}

#ifdef _USE_SIMD
void Y_gate_parallel_simd(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    // const CTYPE imag = 1.i;
    __m256d minus_even = _mm256_set_pd(1, -1, 1, -1);
    __m256d minus_odd = _mm256_set_pd(-1, 1, -1, 1);
    __m256d minus_half = _mm256_set_pd(1, -1, -1, 1);
    if (target_qubit_index == 0) {
        ITYPE basis_index = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (basis_index = 0; basis_index < dim; basis_index += 2) {
            double* ptr0 = (double*)(state + basis_index);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            data0 = _mm256_permute4x64_pd(data0,
                27);  // (3210) -> (0123) : 16+4*2+3=27
            data0 = _mm256_mul_pd(data0, minus_half);
            _mm256_storeu_pd(ptr0, data0);
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_index_1 = basis_index_0 + mask;
            double* ptr0 = (double*)(state + basis_index_0);
            double* ptr1 = (double*)(state + basis_index_1);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            __m256d data1 = _mm256_loadu_pd(ptr1);
            data0 = _mm256_permute_pd(data0, 5);  // (3210) -> (2301) : 4+1
            data1 = _mm256_permute_pd(data1, 5);
            data0 = _mm256_mul_pd(data0, minus_even);
            data1 = _mm256_mul_pd(data1, minus_odd);
            _mm256_storeu_pd(ptr1, data0);
            _mm256_storeu_pd(ptr0, data1);
        }
    }
}
#endif

#ifdef _USE_SVE
void Y_gate_parallel_sve(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;

    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;

    if (mask >= VL) {
#pragma omp parallel
        {
            svbool_t pall = svptrue_b64();
            svfloat64_t sv_minus_even = svzip1(svdup_f64(1.0), svdup_f64(-1.0));
            svfloat64_t sv_minus_odd = svzip1(svdup_f64(-1.0), svdup_f64(1.0));

#pragma omp for
            for (state_index = 0; state_index < loop_dim; state_index += VL) {
                svfloat64_t sv_input0, sv_input1, sv_output0, sv_output1,
                    sv_cval_real, sv_cval_imag;
                ITYPE basis_index_0 =
                    (state_index & mask_low) + ((state_index & mask_high) << 1);
                ITYPE basis_index_1 = basis_index_0 + mask;

                sv_input0 = svld1(pall, (double*)&state[basis_index_0]);
                sv_input1 = svld1(pall, (double*)&state[basis_index_1]);

                sv_cval_real = svuzp1(sv_input0, sv_input1);
                sv_cval_imag = svuzp2(sv_input0, sv_input1);

                sv_output0 = svzip1(sv_cval_imag, sv_cval_real);
                sv_output1 = svzip2(sv_cval_imag, sv_cval_real);

                sv_output0 = svmul_x(pall, sv_output0, sv_minus_odd);
                sv_output1 = svmul_x(pall, sv_output1, sv_minus_even);

                svst1(pall, (double*)&state[basis_index_0], sv_output1);
                svst1(pall, (double*)&state[basis_index_1], sv_output0);
            }
        }
    } else if (dim >= (VL << 1)) {
#pragma omp parallel
        {
            svbool_t pall = svptrue_b64();
            svfloat64_t sv_minus_even = svzip1(svdup_f64(1.0), svdup_f64(-1.0));
            svfloat64_t sv_minus_odd = svzip1(svdup_f64(-1.0), svdup_f64(1.0));
            svfloat64_t sv_minus_half;
            svuint64_t sv_shuffle_table;

            sv_minus_half = svdup_f64(0.0);
            ITYPE len = 0;
            while (len < (VL << 1)) {
                for (ITYPE i = 0; i < (1ULL << target_qubit_index); ++i)
                    sv_minus_half = svext(sv_minus_half, sv_minus_even, 2);
                len += (1ULL << (target_qubit_index + 1));

                for (ITYPE i = 0; i < (1ULL << target_qubit_index); ++i)
                    sv_minus_half = svext(sv_minus_half, sv_minus_odd, 2);
                len += (1ULL << (target_qubit_index + 1));
            }

            sv_shuffle_table = sveor_z(pall, svindex_u64(0, 1),
                svdup_u64(1ULL << (target_qubit_index + 1)));

#pragma omp for
            for (state_index = 0; state_index < dim; state_index += (VL << 1)) {
                svfloat64_t sv_input0, sv_input1, sv_output0, sv_output1,
                    sv_cval_real, sv_cval_imag, sv_shuffle0, sv_shuffle1;

                sv_input0 = svld1(pall, (double*)&state[state_index]);
                sv_input1 = svld1(pall, (double*)&state[state_index + VL]);

                sv_shuffle0 = svtbl(sv_input0, sv_shuffle_table);
                sv_shuffle1 = svtbl(sv_input1, sv_shuffle_table);

                sv_cval_real = svuzp1(sv_shuffle0, sv_shuffle1);
                sv_cval_imag = svuzp2(sv_shuffle0, sv_shuffle1);

                sv_output0 = svzip1(sv_cval_imag, sv_cval_real);
                sv_output1 = svzip2(sv_cval_imag, sv_cval_real);

                sv_shuffle0 = svmul_x(pall, sv_output0, sv_minus_half);
                sv_shuffle1 = svmul_x(pall, sv_output1, sv_minus_half);

                svst1(pall, (double*)&state[state_index], sv_shuffle0);
                svst1(pall, (double*)&state[state_index + VL], sv_shuffle1);
            }
        }
    } else {
        Y_gate_parallel_unroll(target_qubit_index, state, dim);
    }
}
#endif

#ifdef _USE_MPI
void Y_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        Y_gate(target_qubit_index, state, dim);
    } else {
        MPIutil& m = MPIutil::get_inst();
        const int rank = m.get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* t = m.get_workarea(&dim_work, &num_work);
        assert(num_work > 0);
#ifdef _OPENMP
        OMPutil::get_inst().set_qulacs_num_threads(dim_work, 13);
#endif
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        const int pair_rank = rank ^ pair_rank_bit;
        const CTYPE imag = 1.i;
        CTYPE* si = state;
        // printf("#debug dim,dim_work,num_work,t: %lld, %lld, %lld, %p\n", dim,
        // dim_work, num_work, t);
        for (ITYPE iter = 0; iter < num_work; ++iter) {
            m.m_DC_sendrecv(si, t, dim_work, pair_rank);
            ITYPE state_index = 0;
            if (rank & pair_rank_bit) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (state_index = 0; state_index < dim_work; ++state_index) {
                    si[state_index] = imag * t[state_index];
                }
            } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (state_index = 0; state_index < dim_work; ++state_index) {
                    si[state_index] = -imag * t[state_index];
                }
            }
            si += dim_work;
        }

#ifdef _OPENMP
        OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    }
}
#endif
