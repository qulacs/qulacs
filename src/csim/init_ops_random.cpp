#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "MPIutil.hpp"
#include "csim/utility.hpp"
#include "init_ops.hpp"

// state randomization
unsigned long xor_shift(unsigned long* state);
double random_uniform(unsigned long* state);
double random_normal(unsigned long* state);
void initialize_Haar_random_state_with_seed_parallel(
    CTYPE* state, ITYPE dim, UINT outer_qc, UINT seed);

void initialize_Haar_random_state(CTYPE* state, ITYPE dim) {
    initialize_Haar_random_state_with_seed(state, dim, (unsigned)time(NULL));
}

void initialize_Haar_random_state_with_seed(
    CTYPE* state, ITYPE dim, UINT seed) {
    initialize_Haar_random_state_mpi_with_seed(state, dim, 0, seed);
}

void initialize_Haar_random_state_mpi_with_seed(
    CTYPE* state, ITYPE dim, UINT outer_qc, UINT seed) {
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 8);
#endif
    initialize_Haar_random_state_with_seed_parallel(state, dim, outer_qc, seed);
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

void initialize_Haar_random_state_with_seed_parallel(
    CTYPE* state, ITYPE dim, UINT outer_qc, UINT seed) {
#ifndef _USE_MPI  // unused parameter if MPI is not used
    (void)outer_qc;
#endif
    // multi thread
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(dim, 10);
    const UINT thread_count = omp_get_max_threads();
#else
    const UINT thread_count = 1;
#endif
    const int ignore_first = 40;
    const ITYPE block_size = dim / thread_count;
    const ITYPE residual = dim % thread_count;

    unsigned long* random_state_list =
        (unsigned long*)malloc(sizeof(unsigned long) * 4 * thread_count);
    srand(seed);
    for (UINT i = 0; i < 4 * thread_count; ++i) {
        random_state_list[i] = rand();
    }

    double* norm_list = (double*)malloc(sizeof(double) * thread_count);
    for (UINT i = 0; i < thread_count; ++i) {
        norm_list[i] = 0;
    }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
        UINT thread_id = omp_get_thread_num();
#else
        UINT thread_id = 0;
#endif
        unsigned long* my_random_state = random_state_list + 4 * thread_id;
        ITYPE start_index = block_size * thread_id +
                            (residual > thread_id ? thread_id : residual);
        ITYPE end_index =
            block_size * (thread_id + 1) +
            (residual > (thread_id + 1) ? (thread_id + 1) : residual);
        ITYPE index;

        // ignore first randoms
        for (int i = 0; i < ignore_first; ++i) xor_shift(my_random_state);

        for (index = start_index; index < end_index; ++index) {
            double r1, r2;
            r1 = random_normal(my_random_state);
            r2 = random_normal(my_random_state);
            state[index] = r1 + 1.i * r2;
            norm_list[thread_id] += r1 * r1 + r2 * r2;
        }
    }

    double normalizer = 0.;
    for (UINT i = 0; i < thread_count; ++i) {
        normalizer += norm_list[i];
    }
#ifdef _USE_MPI
    if (outer_qc > 0) {
        MPIutil& m = MPIutil::get_inst();
        m.s_D_allreduce(&normalizer);
    }
#endif
    normalizer = 1. / sqrt(normalizer);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (ITYPE index = 0; index < dim; ++index) {
        state[index] *= normalizer;
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    free(random_state_list);
    free(norm_list);
}

unsigned long xor_shift(unsigned long* state) {
    unsigned long t;
    t = (state[0] ^ (state[0] << 11));
    state[0] = state[1];
    state[1] = state[2];
    state[2] = state[3];
    return (state[3] = (state[3] ^ (state[3] >> 19)) ^ (t ^ (t >> 8)));
}
double random_uniform(unsigned long* state) {
    return xor_shift(state) / ((double)ULONG_MAX + 1);
}
double random_normal(unsigned long* state) {
    return sqrt(-1.0 * log(1 - random_uniform(state))) *
           sin(2.0 * M_PI * random_uniform(state));
}
