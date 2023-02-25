#include "stat_ops.hpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "MPIutil.hpp"
#include "constant.hpp"
#include "utility.hpp"

// calculate norm
double state_norm_squared(const CTYPE* state, ITYPE dim) {
    ITYPE index;
    double norm = 0;

#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
#pragma omp parallel for reduction(+ : norm)
#endif
    for (index = 0; index < dim; ++index) {
        norm += pow(_cabs(state[index]), 2);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif

    return norm;
}

// calculate norm
double state_norm_squared_single_thread(const CTYPE* state, ITYPE dim) {
    ITYPE index;
    double norm = 0;
    for (index = 0; index < dim; ++index) {
        norm += pow(_cabs(state[index]), 2);
    }
    return norm;
}

// calculate norm for mpi
#ifdef _USE_MPI
double state_norm_squared_mpi(const CTYPE* state, ITYPE dim) {
    ITYPE index;
    double norm = 0;

#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 15);
#pragma omp parallel for reduction(+ : norm)
#endif
    for (index = 0; index < dim; ++index) {
        norm += pow(_cabs(state[index]), 2);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif

    MPIutil::get_inst().s_D_allreduce(&norm);

    return norm;
}
#endif

// calculate inner product of two state vector
CTYPE
state_inner_product(const CTYPE* state_bra, const CTYPE* state_ket, ITYPE dim) {
    double real_sum = 0.;
    double imag_sum = 0.;
    ITYPE index;

#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 15);
#pragma omp parallel for reduction(+ : real_sum, imag_sum)
#endif
    for (index = 0; index < dim; ++index) {
        CTYPE value;
        value += conj(state_bra[index]) * state_ket[index];
        real_sum += _creal(value);
        imag_sum += _cimag(value);
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif

    return real_sum + 1.i * imag_sum;
}

#ifdef _USE_MPI
// calculate inner product of two state vector for mpi
CTYPE
state_inner_product_mpi(const CTYPE* state_bra, const CTYPE* state_ket,
    ITYPE dim_bra, ITYPE dim_ket) {
    CTYPE sum = 0.;
    ITYPE index;
    ITYPE dim = std::min(dim_bra, dim_ket);
    MPIutil& m = MPIutil::get_inst();
    const CTYPE* new_bra = state_bra;
    const CTYPE* new_ket = state_ket;
    ITYPE offs = m.get_rank() * dim;
    if (dim_bra < dim_ket)
        new_ket += offs;
    else if (dim_bra > dim_ket)
        new_bra += offs;

    sum = state_inner_product(new_bra, new_ket, dim);
    m.s_DC_allreduce(&sum);
    return sum;
}
#endif

void state_tensor_product(const CTYPE* state_left, ITYPE dim_left,
    const CTYPE* state_right, ITYPE dim_right, CTYPE* state_dst) {
    ITYPE index_left, index_right;
    for (index_left = 0; index_left < dim_left; ++index_left) {
        CTYPE val_left = state_left[index_left];
        for (index_right = 0; index_right < dim_right; ++index_right) {
            state_dst[index_left * dim_right + index_right] =
                val_left * state_right[index_right];
        }
    }
}
void state_permutate_qubit(const UINT* qubit_order, const CTYPE* state_src,
    CTYPE* state_dst, UINT qubit_count, ITYPE dim) {
    ITYPE index;
    for (index = 0; index < dim; ++index) {
        ITYPE src_index = 0;
        for (UINT qubit_index = 0; qubit_index < qubit_count; ++qubit_index) {
            if ((index >> qubit_index) % 2) {
                src_index += 1ULL << qubit_order[qubit_index];
            }
        }
        state_dst[index] = state_src[src_index];
    }
}

void state_drop_qubits(const UINT* target, const UINT* projection,
    UINT target_count, const CTYPE* state_src, CTYPE* state_dst, ITYPE dim) {
    ITYPE dst_dim = dim >> target_count;
    UINT* sorted_target = create_sorted_ui_list(target, target_count);
    ITYPE projection_mask = 0;
    for (UINT target_index = 0; target_index < target_count; ++target_index) {
        projection_mask ^= (projection[target_index] << target[target_index]);
    }

    ITYPE index;
    for (index = 0; index < dst_dim; ++index) {
        ITYPE src_index = index;
        for (UINT target_index = 0; target_index < target_count;
             ++target_index) {
            UINT insert_index = sorted_target[target_index];
            src_index = insert_zero_to_basis_index(
                src_index, 1ULL << insert_index, insert_index);
        }
        src_index ^= projection_mask;
        state_dst[index] = state_src[src_index];
    }
    free(sorted_target);
}
