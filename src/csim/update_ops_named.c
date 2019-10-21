
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



void P0_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    ITYPE state_index;
    ITYPE mask = (1ULL << target_qubit_index);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0 ; state_index<loop_dim ; ++state_index){
        ITYPE temp_index = insert_zero_to_basis_index(state_index,mask,target_qubit_index) ^ mask;
        state[temp_index] = 0;
    }
}

void P1_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim){
    const ITYPE loop_dim = dim/2;
    ITYPE mask = (1ULL << target_qubit_index);
    ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0 ; state_index<loop_dim ; ++state_index){
        ITYPE temp_index = insert_zero_to_basis_index(state_index,mask,target_qubit_index);
        state[temp_index] = 0;
    }
}

void normalize(double norm, CTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim;
    const double normalize_factor = sqrt(1./norm);
    ITYPE state_index;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(state_index=0 ; state_index<loop_dim ; ++state_index){
        state[state_index] *= normalize_factor;
    }
}

void RX_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim){
    single_qubit_Pauli_rotation_gate(target_qubit_index, 1, angle, state, dim);
}

void RY_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim){
    single_qubit_Pauli_rotation_gate(target_qubit_index, 2, angle, state, dim);
}

void RZ_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim){
    single_qubit_Pauli_rotation_gate(target_qubit_index, 3, angle, state, dim);
}

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
