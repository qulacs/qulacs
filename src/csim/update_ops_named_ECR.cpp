

#include <cstring>
#include <bitset>
#include <complex>
#include <iostream>
#include <vector>
using namespace std::complex_literals;
#include <algorithm>


#include "MPIutil.hpp"
#include "constant.hpp"
#include "update_ops.hpp"
#include "utility.hpp"
#include "csim/type.hpp"
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


void ECR_gate(UINT target_qubit_index_0, UINT target_qubit_index_1,
    CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    ECR_gate_parallel_simd(
        target_qubit_index_0, target_qubit_index_1, state, dim);
#elif defined(_USE_SVE)
    ECR_gate_parallel_sve(
        target_qubit_index_0, target_qubit_index_1, state, dim);
#else
    ECR_gate_parallel_unroll(
        target_qubit_index_0, target_qubit_index_1, state, dim);
#endif

#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}


void ECR_gate_parallel_unroll(UINT target_qubit_index_0,
    UINT target_qubit_index_1, CTYPE* state, ITYPE dim) {
    
    const ITYPE loop_dim = dim / 4;

    const ITYPE mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE mask_1 = 1ULL << target_qubit_index_1;
    const ITYPE mask = mask_0 + mask_1;

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index_0, target_qubit_index_1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const double sqrt2inv = 1. / sqrt(2.);


#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (ITYPE state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_index_00 = (state_index & low_mask) +
                               ((state_index & mid_mask) << 1) +
                               ((state_index & high_mask) << 2);
        ITYPE basis_index_01 = basis_index_00 + mask_0;
        ITYPE basis_index_10 = basis_index_00 + mask_1;
        ITYPE basis_index_11 = basis_index_00 + mask;

        CTYPE v00 = state[basis_index_00];
        CTYPE v01 = state[basis_index_01];
        CTYPE v10 = state[basis_index_10];
        CTYPE v11 = state[basis_index_11];

        CTYPE new_v00 = sqrt2inv * (v01 + 1.i * v11);
        CTYPE new_v01 = sqrt2inv * (v00 - 1.i * v10);
        CTYPE new_v10 = sqrt2inv * (v11 + 1.i * v01);
        CTYPE new_v11 = sqrt2inv * (v10 - 1.i * v00);

        state[basis_index_00] = new_v00;
        state[basis_index_01] = new_v01;
        state[basis_index_10] = new_v10;
        state[basis_index_11] = new_v11;
    }

} 

#ifdef _USE_SIMD
void ECR_gate_parallel_simd(UINT target_qubit_index_0,
    UINT target_qubit_index_1, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE mask_1 = 1ULL << target_qubit_index_1;
    const ITYPE mask = mask_0 + mask_1;

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index_0, target_qubit_index_1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);
    const double sqrt2inv = 1. / sqrt(2.);

    if (min_qubit_index == 0) {
        ECR_gate_parallel_unroll(target_qubit_index_0,
    target_qubit_index_1, state, dim);
    } else {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
        for (ITYPE state_index = 0; state_index < loop_dim; state_index+=2) {
            ITYPE basis_index_00 = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2);
            ITYPE basis_index_01 = basis_index_00 + mask_0;
            ITYPE basis_index_10 = basis_index_00 + mask_1;
            ITYPE basis_index_11 = basis_index_00 + mask;

            double* ptr00 = (double*)(state + basis_index_00);
            double* ptr01 = (double*)(state + basis_index_01);
            double* ptr10 = (double*)(state + basis_index_10);
            double* ptr11 = (double*)(state + basis_index_11);

            __m256d a_lo = _mm256_loadu_pd(ptr00); 
            __m256d a_hi = _mm256_loadu_pd(ptr01); 
            __m256d b_lo = _mm256_loadu_pd(ptr10); 
            __m256d b_hi = _mm256_loadu_pd(ptr11); 

            auto mul_by_i_256 = [](const __m256d& x) -> __m256d {
                __m256d swapped = _mm256_permute_pd(x, 0b0101); // swap Re <-> Im
                const __m256d sign = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
                return _mm256_mul_pd(swapped, sign);
            };

            __m256d i_b_hi = mul_by_i_256(b_hi);
            __m256d i_b_lo = mul_by_i_256(b_lo);
            __m256d i_a_hi = mul_by_i_256(a_hi);
            __m256d i_a_lo = mul_by_i_256(a_lo);

            __m256d tmp_new_v00 = _mm256_add_pd(a_hi, i_b_hi);
            __m256d tmp_new_v01 = _mm256_sub_pd(a_lo, i_b_lo);
            __m256d tmp_new_v10 = _mm256_add_pd(b_hi, i_a_hi);
            __m256d tmp_new_v11 = _mm256_sub_pd(b_lo, i_a_lo);

            __m256d svec = _mm256_set1_pd(sqrt2inv);
            tmp_new_v00 = _mm256_mul_pd(tmp_new_v00, svec);
            tmp_new_v01 = _mm256_mul_pd(tmp_new_v01, svec);
            tmp_new_v10 = _mm256_mul_pd(tmp_new_v10, svec);
            tmp_new_v11 = _mm256_mul_pd(tmp_new_v11, svec);

            _mm256_storeu_pd(ptr00, tmp_new_v00); 
            _mm256_storeu_pd(ptr01, tmp_new_v01); 
            _mm256_storeu_pd(ptr10, tmp_new_v10); 
            _mm256_storeu_pd(ptr11, tmp_new_v11); 
        }
    }
} 
#endif


#ifdef _USE_SVE

static inline svfloat64_t mul_by_i(svbool_t pg, svfloat64_t x) {
    svuint64_t tbl_idx = svindex_u64(0, 1);      
    tbl_idx = sveor_z(pg, tbl_idx, svdup_u64(1));

    svfloat64_t swapped = svtbl_f64(x, tbl_idx);

    svbool_t odd = svcmpne(pg, svand_z(pg, tbl_idx, svdup_u64(1)), svdup_u64(0));

    svfloat64_t sign = svsel(odd, svdup_f64(-1.0), svdup_f64(1.0));

    return svmul_x(pg, swapped, sign);
}


void ECR_gate_parallel_sve(UINT target_qubit_index_0,
                           UINT target_qubit_index_1,
                           CTYPE* state, ITYPE dim) {

    const ITYPE loop_dim = dim / 4;
    const ITYPE mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE mask_1 = 1ULL << target_qubit_index_1;
    const ITYPE mask = mask_0 + mask_1;

    const UINT min_qubit_index = get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index = get_max_ui(target_qubit_index_0, target_qubit_index_1);

    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask  = min_qubit_mask - 1;
    const ITYPE mid_mask  = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);
    const double sqrt2inv = 1. / sqrt(2.);

    // # of complex128 numbers in an SVE register
    ITYPE VL = svcntd() / 2;


    if ((dim > VL) && (min_qubit_mask >= VL)) {

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
        for (ITYPE state_index = 0; state_index < loop_dim; state_index+=VL) {

            ITYPE basis_index_00 = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2);
            ITYPE basis_index_01 = basis_index_00 + mask_0;
            ITYPE basis_index_10 = basis_index_00 + mask_1;
            ITYPE basis_index_11 = basis_index_00 + mask;


            svfloat64_t input00 = svld1(svptrue_b64(), (double*)&state[basis_index_00]);
            svfloat64_t input01 = svld1(svptrue_b64(), (double*)&state[basis_index_01]);
            svfloat64_t input10 = svld1(svptrue_b64(), (double*)&state[basis_index_10]);
            svfloat64_t input11 = svld1(svptrue_b64(), (double*)&state[basis_index_11]);


            svfloat64_t i_00 = mul_by_i(svptrue_b64(),  input00);
            svfloat64_t i_10 = mul_by_i(svptrue_b64(),  input10);
            svfloat64_t i_11 = mul_by_i(svptrue_b64(),  input11);
            svfloat64_t i_01 = mul_by_i(svptrue_b64(),  input01);


            svfloat64_t output00 = svadd_x(svptrue_b64(), input01, i_11); 
            svfloat64_t output01 = svsub_x(svptrue_b64(), input00, i_10); 
            svfloat64_t output10 = svadd_x(svptrue_b64(), input11, i_01); 
            svfloat64_t output11 = svsub_x(svptrue_b64(), input10, i_00); 


            svfloat64_t sv_factor = svdup_f64(sqrt2inv);

            output00 = svmul_x(svptrue_b64(), output00, sv_factor);
            output01 = svmul_x(svptrue_b64(), output01, sv_factor);
            output10 = svmul_x(svptrue_b64(), output10, sv_factor);
            output11 = svmul_x(svptrue_b64(), output11, sv_factor);

            svst1(svptrue_b64(), (double *)&state[basis_index_00], output00);
            svst1(svptrue_b64(), (double *)&state[basis_index_01], output01);
            svst1(svptrue_b64(), (double *)&state[basis_index_10], output10);
            svst1(svptrue_b64(), (double *)&state[basis_index_11], output11);

            
        }
    }

    else {
        ECR_gate_parallel_unroll(target_qubit_index_0, target_qubit_index_1, state, dim);
    }

}

#endif  // _USE_SVE

#ifdef _USE_MPI
#include <bitset>

void ECR_gate_mpi(UINT target_qubit_index_0, UINT target_qubit_index_1,
    CTYPE* state, ITYPE dim, UINT inner_qc) {
    UINT left_qubit, right_qubit;
    if (target_qubit_index_0 > target_qubit_index_1) {
        left_qubit = target_qubit_index_0;
        right_qubit = target_qubit_index_1;
    } else {
        left_qubit = target_qubit_index_1;
        right_qubit = target_qubit_index_0;
    }

    if (left_qubit < inner_qc) {
        ECR_gate(target_qubit_index_0, target_qubit_index_1, state, dim);

    } else if (right_qubit < inner_qc) {  // one target is outer
        MPIutil& m = MPIutil::get_inst();
        const UINT rank = m.get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* t = m.get_workarea(&dim_work, &num_work);
        const ITYPE tgt_rank_bit = 1 << (left_qubit - inner_qc);
        const ITYPE rtgt_blk_dim = 1 << right_qubit;
        const int pair_rank = rank ^ tgt_rank_bit;
        bool is_lower_rank = !(rank & tgt_rank_bit);
        CTYPE* si = state;

        for (UINT i = 0; i < (UINT)num_work; ++i) {
            m.m_DC_sendrecv(si, t, dim_work, pair_rank);
            _ECR_gate_mpi_local_global(target_qubit_index_0, target_qubit_index_1, t, si, dim_work, rtgt_blk_dim, is_lower_rank);
            si += dim_work;
        }

    } else {  // both targets are outer
        MPIutil& m = MPIutil::get_inst();
        const UINT rank = m.get_rank();
        int world_size_int = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_int); 
        const int world_size = world_size_int; 
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        const UINT tgt0_rank_bit = 1 << (target_qubit_index_0 - inner_qc);  
        const UINT tgt1_rank_bit = 1 << (target_qubit_index_1 - inner_qc);  
        const UINT tgt_rank_bit = tgt0_rank_bit + tgt1_rank_bit; 
        const int pair_rank = rank ^ tgt0_rank_bit;  
        const int pair_rank1 = rank ^ tgt_rank_bit;  
        CTYPE* tmp = m.get_workarea(
            &dim_work, &num_work); 
        (void)tmp;            
        std::vector<CTYPE> t1_buf(dim_work);
        std::vector<CTYPE> t2_buf(dim_work);
        CTYPE* t1 = t1_buf.data();
        CTYPE* t2 = t2_buf.data();
        CTYPE* si = state; 

        for (UINT i = 0; i < (UINT)num_work; ++i) {
            CTYPE* si_i = state + i * dim_work; 
            m.m_DC_sendrecv(si_i, t1, dim_work, pair_rank);
        }

        for (UINT j = 0; j < (UINT)num_work; ++j) {
            CTYPE* si_j = state + j * dim_work;  
            m.m_DC_sendrecv(si_j, t2, dim_work, pair_rank1);
        }

        const UINT local_qc = (UINT)std::log2(dim);
        bool is_lower_rank = !(rank & tgt0_rank_bit);
        _ECR_gate_mpi_external(t1, t2, si, dim_work, is_lower_rank);
        si += dim_work;
    }
}

void _ECR_gate_mpi_local_global(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE* t, CTYPE* si, ITYPE dim, ITYPE rtgt_blk_dim, bool is_lower_rank) {
    const double sqrt2inv = 1. / sqrt(2.);
    const ITYPE amplitude_block_size = rtgt_blk_dim << 1; 

#pragma omp parallel for
    for (ITYPE k = 0; k < dim/2; ++k) {
            const ITYPE state_index = k/rtgt_blk_dim * amplitude_block_size;
            const ITYPE offset = k%rtgt_blk_dim;
            const ITYPE idx0 = state_index + offset; 
            const ITYPE idx1 = idx0 + rtgt_blk_dim; 
            if (target_qubit_index_0 < target_qubit_index_1) {
                const CTYPE si0 = si[idx0];
                si[idx0] = (si[idx1] + t[idx1] * 1i) * sqrt2inv;
                si[idx1] = (si0 - t[idx0] * 1i) * sqrt2inv;
            } else {
                if (is_lower_rank) {
                    si[idx0] = ( t[idx0] + t[idx1] * 1i) * sqrt2inv; 
                    si[idx1] = ( t[idx0] * 1i + t[idx1]) * sqrt2inv;   
                } else {
                    si[idx0] = ( t[idx0] - t[idx1] * 1i) * sqrt2inv;
                    si[idx1] = (-t[idx0] * 1i + t[idx1]) * sqrt2inv;
                }
            }
    }  
}

void _ECR_gate_mpi_external(
    CTYPE* t1, CTYPE* t2, CTYPE* si, ITYPE dim, bool is_lower_rank) {
    const double sqrt2inv = 1. / sqrt(2.);
    ITYPE state_index = 0;
    #pragma omp parallel for
        for (ITYPE i = 0; i < dim; ++i) {
            if (is_lower_rank) {
                si[i] = (t1[i] + t2[i] * 1i) * sqrt2inv;
            } else {
                si[i] = (t1[i] - t2[i] * 1i) * sqrt2inv;
            }
        }
}

#endif
