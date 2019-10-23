
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "constant.h"
#include "update_ops.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void CUz_gate(double angle, UINT c_bit, UINT t_bit, CTYPE *psi, ITYPE dim)
{
    //ITYPE i;
    ITYPE j;
    ITYPE head, body, tail;
    ITYPE basis00, basis11;
    ITYPE left, right;
    //CTYPE U_gate[2][2];
    /*
    for(i = 0; i < 2; ++i)
        for(j = 0; j < 2; ++j)
            U_gate[i][j] = cos(angle) * PAULI_MATRIX[0][i][j] + sin(angle) * 1.0i * PAULI_MATRIX[3][i][j];
    */

    if (t_bit > c_bit){
        left = t_bit;
        right = c_bit;
    } else {
        left = c_bit;
        right = t_bit;
    }

#ifdef _OPENMP
#pragma omp parallel for private(head, body, tail, basis00, basis11)
#endif
    for(j = 0; j < dim/4; ++j) {
        head = j >> (left - 1);
        body = (j & ((1ULL << (left - 1)) - 1)) >> right; // (j % 2^(k-1)) >> i
        tail = j & ((1ULL << right) - 1); // j%(2^i)

        basis00=(head<<(left+1))+(body<<(right+1))+tail;
        basis11=basis00+(1ULL<<c_bit)+(1ULL<<t_bit);
        
        psi[basis11] = cos(angle) * psi[basis11] + sin(angle) * 1.0i * psi[basis11];
        
    }

}

// qft: apply qft from k th to k+Nbits th
void qft(UINT k, UINT Nbits, int doSWAP, CTYPE *psi, ITYPE dim){
    UINT i, j;
    for(i=1;i<Nbits;++i){
        single_qubit_dense_matrix_gate(Nbits-i+k, HADAMARD_MATRIX, psi, dim);
        // printf("Had %d\n",Nbits-i+k);
        for(j=0;j<i;++j){
            // printf("CUZ %d %d %.5f*PI\n", Nbits-i-1+k, Nbits-j-1+k, 1.0/(1ULL<<(i-j)));
            //CUz_gate(1.0*PI/(1ULL<<(i-j)), Nbits-i-1+k, Nbits-j-1+k, psi);
            CUz_gate(1.0*PI/(1ULL<<(i-j)), Nbits-i-1+k, Nbits-j-1+k, psi, dim);
        }
    }
    single_qubit_dense_matrix_gate(k,HADAMARD_MATRIX, psi, dim);
    // printf("Had %d\n",k);
    if(doSWAP){
        for(i=0;i<Nbits/2;++i){
            SWAP_gate(i+k,Nbits-1-i+k,psi, dim);
            // printf("SWAP %d %d\n",i+k,Nbits-1-i+k);
        }
    }
}

// inverse_qft: apply inverse_qft from k th to k+Nbits th
void inverse_qft(UINT k, UINT Nbits, int doSWAP, CTYPE *psi, ITYPE dim){
    UINT i, j;
    if(doSWAP){
        for(i=0;i<Nbits/2;++i){
            SWAP_gate(i+k,Nbits-1-i+k,psi, dim);
            // printf("SWAP %d %d\n",i+k,Nbits-1-i+k);
        }
    }
    for(i=0;i<Nbits;++i){
        single_qubit_dense_matrix_gate(i+k,HADAMARD_MATRIX, psi,dim);
        // printf("Had %d\n",i+k);
        for(j=i+1;j<Nbits;++j){
            // printf("CUZ %d %d %.5f*PI\n",i+k,j+k,-1.0/(1ULL<<(j-i)));
            // CUz_gate(-1.0*PI/(1ULL<<(j-i)), i+k, j+k, psi);
            CUz_gate(-1.0*PI/(1ULL<<(j-i)), i+k, j+k, psi, dim);
        }
    }
}
