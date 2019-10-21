
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "constant.h"
#include "update_ops.h"
#include "utility.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void single_qubit_Pauli_gate(UINT target_qubit_index, UINT Pauli_operator_type, CTYPE *state, ITYPE dim) {
    switch(Pauli_operator_type){
    case 0:
        break;
    case 1:
        X_gate(target_qubit_index,state,dim);
        break;
    case 2:
        Y_gate(target_qubit_index,state,dim);
        break;
    case 3:
        Z_gate(target_qubit_index,state,dim);
        break;
    default:
        fprintf(stderr,"invalid Pauli operation is called");
        assert(0);
    }
}


void single_qubit_Pauli_rotation_gate(UINT target_qubit_index, UINT Pauli_operator_index, double angle, CTYPE *state, ITYPE dim) {
    // create matrix and call dense matrix
    UINT i, j;
    CTYPE rotation_gate[4];
    for(i = 0; i < 2; ++i)
        for(j = 0; j < 2; ++j)
            rotation_gate[i*2+j] = cos(angle/2) * PAULI_MATRIX[0][i*2+j] + sin(angle/2) * 1.0i * PAULI_MATRIX[Pauli_operator_index][i*2+j];

    single_qubit_dense_matrix_gate(target_qubit_index, rotation_gate, state, dim);
}
