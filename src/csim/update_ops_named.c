
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "constant.h"
#include "update_ops.h"
#include "utility.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

void S_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_phase_gate(target_qubit_index, 1.i, state, dim);
}
void Sdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_phase_gate(target_qubit_index, -1.i, state, dim);
}
void T_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_phase_gate(target_qubit_index, (1.+1.i)/sqrt(2.), state, dim);
}
void Tdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_phase_gate(target_qubit_index, (1.-1.i)/sqrt(2.), state, dim);
}
void sqrtX_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_X_GATE_MATRIX, state, dim);
}
void sqrtXdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_X_DAG_GATE_MATRIX, state, dim);
}
void sqrtY_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_Y_GATE_MATRIX, state, dim);
}
void sqrtYdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim){
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_Y_DAG_GATE_MATRIX, state, dim);
}
