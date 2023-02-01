
#include "MPIutil.hpp"
#include "update_ops.hpp"
#include "utility.hpp"

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void H_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    H_gate_parallel_simd(target_qubit_index, state, dim);
#else
    H_gate_parallel_unroll(target_qubit_index, state, dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void H_gate_parallel_unroll(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    const double sqrt2inv = 1. / sqrt(2.);
    ITYPE state_index = 0;
    if (target_qubit_index == 0) {
        ITYPE basis_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (basis_index = 0; basis_index < dim; basis_index += 2) {
            CTYPE temp0 = state[basis_index];
            CTYPE temp1 = state[basis_index + 1];
            state[basis_index] = (temp0 + temp1) * sqrt2inv;
            state[basis_index + 1] = (temp0 - temp1) * sqrt2inv;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_index_1 = basis_index_0 + mask;
            CTYPE temp_a0 = state[basis_index_0];
            CTYPE temp_a1 = state[basis_index_1];
            CTYPE temp_b0 = state[basis_index_0 + 1];
            CTYPE temp_b1 = state[basis_index_1 + 1];
            state[basis_index_0] = (temp_a0 + temp_a1) * sqrt2inv;
            state[basis_index_1] = (temp_a0 - temp_a1) * sqrt2inv;
            state[basis_index_0 + 1] = (temp_b0 + temp_b1) * sqrt2inv;
            state[basis_index_1 + 1] = (temp_b0 - temp_b1) * sqrt2inv;
        }
    }
}

#ifdef _USE_SIMD
void H_gate_parallel_simd(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    // const CTYPE imag = 1.i;
    const double sqrt2inv = 1. / sqrt(2.);
    __m256d sqrt2inv_array =
        _mm256_set_pd(sqrt2inv, sqrt2inv, sqrt2inv, sqrt2inv);
    if (target_qubit_index == 0) {
        //__m256d sqrt2inv_array_half = _mm256_set_pd(sqrt2inv, sqrt2inv,
        //-sqrt2inv, -sqrt2inv);
        ITYPE basis_index = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (basis_index = 0; basis_index < dim; basis_index += 2) {
            double* ptr0 = (double*)(state + basis_index);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            __m256d data1 = _mm256_permute4x64_pd(data0,
                78);  // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
            __m256d data2 = _mm256_add_pd(data0, data1);
            __m256d data3 = _mm256_sub_pd(data1, data0);
            __m256d data4 =
                _mm256_blend_pd(data3, data2, 3);  // take data3 for latter half
            data4 = _mm256_mul_pd(data4, sqrt2inv_array);
            _mm256_storeu_pd(ptr0, data4);
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
            __m256d data2 = _mm256_add_pd(data0, data1);
            __m256d data3 = _mm256_sub_pd(data0, data1);
            data2 = _mm256_mul_pd(data2, sqrt2inv_array);
            data3 = _mm256_mul_pd(data3, sqrt2inv_array);
            _mm256_storeu_pd(ptr0, data2);
            _mm256_storeu_pd(ptr1, data3);
        }
    }
}
#endif

#ifdef _USE_MPI
void H_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        H_gate(target_qubit_index, state, dim);
    } else {
        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* t = m->get_workarea(&dim_work, &num_work);
        assert(num_work > 0);
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        const int pair_rank = rank ^ pair_rank_bit;

#ifdef _OPENMP
		OMPutil::get_inst().set_qulacs_num_threads(dim_work, 13);
#endif

        CTYPE* si = state;
        for (UINT i = 0; i < (UINT)num_work; ++i) {
            m->m_DC_sendrecv(si, t, dim_work, pair_rank);

            _H_gate_mpi(t, si, dim_work, rank & pair_rank_bit);

            si += dim_work;
        }
#ifdef _OPENMP
		OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    }
}

void _H_gate_mpi(CTYPE* t, CTYPE* si, ITYPE dim, int flag) {
    const double sqrt2inv = 1. / sqrt(2.);
    ITYPE state_index = 0;

#pragma omp parallel for
    for (state_index = 0; state_index < dim; state_index += 2) {
        // flag: My qubit(target in outer_qubit) value.
        if (flag) {
            // state-value=0, t-value=1
            si[state_index] = (t[state_index] - si[state_index]) * sqrt2inv;
            si[state_index + 1] =
                (t[state_index + 1] - si[state_index + 1]) * sqrt2inv;
        } else {
            // state-value=1, t-value=0
            si[state_index] = (si[state_index] + t[state_index]) * sqrt2inv;
            si[state_index + 1] =
                (si[state_index + 1] + t[state_index + 1]) * sqrt2inv;
        }
    }
}
#endif  // #ifdef _USE_MPI
