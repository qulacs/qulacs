#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "stat_ops.h"
#include "utility.h"
#include "constant.h"


// calculate norm
double state_norm_squared(const CTYPE *state, ITYPE dim) {
    ITYPE index;
    double norm = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:norm)
#endif
    for (index = 0; index < dim; ++index){
        norm += pow(cabs(state[index]), 2);
    }
    return norm;
}

// calculate inner product of two state vector
CTYPE state_inner_product(const CTYPE *state_bra, const CTYPE *state_ket, ITYPE dim) {
#ifndef _MSC_VER
    CTYPE value = 0;
    ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:value)
#endif
    for(index = 0; index < dim; ++index){
        value += conj(state_bra[index]) * state_ket[index];
    }
    return value;
#else

    double real_sum = 0.;
    double imag_sum = 0.;
    ITYPE index;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:real_sum,imag_sum)
#endif
    for (index = 0; index < dim; ++index) {
        CTYPE value;
        value += conj(state_bra[index]) * state_ket[index];
        real_sum += creal(value);
        imag_sum += cimag(value);
    }
    return real_sum + 1.i * imag_sum;
#endif
}


