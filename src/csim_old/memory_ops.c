#include <stdio.h>
#include <stdlib.h>
#include "memory_ops.h"
#include "utility.h"
#include <time.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// state functions
// allocate quantum state
CTYPE* allocate_quantum_state(ITYPE dim) {
    CTYPE* state = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*dim));

    if (!state){
        fprintf(stderr,"Out of memory\n");
        exit(1);
    }
    return state;
}

// initialize state
void initialize_quantum_state(CTYPE *state, ITYPE dim) {
    ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(index=0; index < dim ; ++index){
        state[index]=0;
    }
    state[0] = 1.0;
}

void release_quantum_state(CTYPE* state){
    free(state);
}




unsigned long xor128(unsigned long* state);
double random_uniform(unsigned long* state);
double random_normal(unsigned long* state);

unsigned long xor128(unsigned long* state){
    unsigned long t;
    t = (state[0] ^ (state[0] << 11));
    state[0] = state[1]; state[1] = state[2]; state[2] = state[3];
    return (state[3] = (state[3] ^ (state[3] >> 19)) ^ (t ^ (t >> 8)));
}
double random_uniform(unsigned long* state) {
    return xor128(state) / ((float)ULONG_MAX);
}
double random_normal(unsigned long* state) {
    return sqrt(-1.0*log(random_uniform(state))) * sin(2.0*M_PI*random_uniform(state));
}


void initialize_Haar_random_state(CTYPE *state, ITYPE dim) {
    initialize_Haar_random_state_with_seed(state, dim, (unsigned)time(NULL));
}

void initialize_Haar_random_state_with_seed(CTYPE *state, ITYPE dim, UINT seed) {
#ifndef _OPENMP
    // single thread
    const int ignore_first = 40;
    double norm = 0.;
    unsigned long random_state[4];
    srand(seed);
    random_state[0] = rand();
    random_state[1] = rand();
    random_state[2] = rand();
    random_state[3] = rand();
    for (int i = 0; i < ignore_first; ++i) xor128(random_state);
    for (ITYPE index = 0; index < dim; ++index) {
        double r1, r2;
        r1 = random_normal(random_state);
        r2 = random_normal(random_state);
        state[index] = r1 + 1.i * r2;
        norm += r1 * r1 + r2 * r2;
    }
    norm = sqrt(norm);
    for (ITYPE index = 0; index < dim; ++index) {
        state[index] /= norm;
    }
#else
    // multi thread
    const int ignore_first = 40;
    const UINT thread_count = omp_get_max_threads();
    const ITYPE block_size = dim / thread_count;
    const ITYPE residual = dim % thread_count;

    unsigned long* random_state_list = (unsigned long*)malloc(sizeof(unsigned long)*4*thread_count);
    srand(seed);
    for (UINT i = 0; i < 4*thread_count; ++i) {
        random_state_list[i] = rand();
    }

    double* norm_list = (double*)malloc(sizeof(double)*thread_count);
    for (UINT i = 0; i < thread_count; ++i) {
        norm_list[i] = 0;
    }
#pragma omp parallel
    {
        UINT thread_id = omp_get_thread_num();
        unsigned long* my_random_state = random_state_list +4 * thread_id;
        ITYPE start_index = block_size * thread_id + (residual>thread_id?thread_id:residual);
        ITYPE end_index = block_size * (thread_id+1) + (residual > (thread_id+1) ? (thread_id+1) : residual);
        ITYPE index;

        // ignore first randoms
        for (int i = 0; i < ignore_first; ++i) xor128(my_random_state);

        for (index = start_index; index < end_index; ++index) {
            double r1, r2;
            r1 = r2 = 1;
            r1 = random_normal(my_random_state);
            r2 = random_normal(my_random_state);
            state[index] = r1 + 1.i * r2;
            norm_list[thread_id] += r1 * r1 + r2 * r2;
        }
    }

    double norm = 0.;
    for (UINT i = 0; i < thread_count; ++i) {
        norm += norm_list[i];
    }
    norm = sqrt(norm);

#pragma omp parallel for
    for (ITYPE index = 0; index < dim; ++index) {
        state[index] /= norm;
    }

    free(random_state_list);
    free(norm_list);
#endif
}
