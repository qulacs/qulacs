
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.hpp"
#include "update_ops.hpp"
#include "utility.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void single_qubit_Pauli_gate(UINT target_qubit_index, UINT Pauli_operator_type,
    CTYPE* state, ITYPE dim) {
    switch (Pauli_operator_type) {
        case 0:
            break;
        case 1:
            X_gate(target_qubit_index, state, dim);
            break;
        case 2:
            Y_gate(target_qubit_index, state, dim);
            break;
        case 3:
            Z_gate(target_qubit_index, state, dim);
            break;
        default:
            fprintf(stderr, "invalid Pauli operation is called");
            assert(0);
    }
}

void single_qubit_Pauli_rotation_gate(UINT target_qubit_index,
    UINT Pauli_operator_index, double angle, CTYPE* state, ITYPE dim) {
    switch (Pauli_operator_index) {
        case 0:
            break;
        case 1:
            RX_gate(target_qubit_index, angle, state, dim);
            break;
        case 2:
            RY_gate(target_qubit_index, angle, state, dim);
            break;
        case 3:
            RZ_gate(target_qubit_index, angle, state, dim);
            break;
        default:
            fprintf(stderr, "invalid Pauli operation is called");
            assert(0);
    }
}

#define _GET_RX_MATRIX_  \
    CTYPE matrix[4];     \
    double c, s;         \
    c = cos(angle / 2);  \
    s = sin(angle / 2);  \
    matrix[0] = c;       \
    matrix[1] = 1.i * s; \
    matrix[2] = 1.i * s; \
    matrix[3] = c;

void RX_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim) {
    _GET_RX_MATRIX_
    single_qubit_dense_matrix_gate(target_qubit_index, matrix, state, dim);
}

#ifdef _USE_MPI
void RX_gate_mpi(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim,
    UINT inner_qc) {
    _GET_RX_MATRIX_
    if (target_qubit_index < inner_qc) {
        single_qubit_dense_matrix_gate(target_qubit_index, matrix, state, dim);
    } else {
        single_qubit_dense_matrix_gate_mpi(
            target_qubit_index, matrix, state, dim, inner_qc);
    }
}
#endif

#define _GET_RY_MATRIX_ \
    CTYPE matrix[4];    \
    double c, s;        \
    c = cos(angle / 2); \
    s = sin(angle / 2); \
    matrix[0] = c;      \
    matrix[1] = s;      \
    matrix[2] = -s;     \
    matrix[3] = c;

void RY_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim) {
    _GET_RY_MATRIX_
    single_qubit_dense_matrix_gate(target_qubit_index, matrix, state, dim);
}

#ifdef _USE_MPI
void RY_gate_mpi(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim,
    UINT inner_qc) {
    _GET_RY_MATRIX_
    if (target_qubit_index < inner_qc) {
        single_qubit_dense_matrix_gate(target_qubit_index, matrix, state, dim);
    } else {
        single_qubit_dense_matrix_gate_mpi(
            target_qubit_index, matrix, state, dim, inner_qc);
    }
}
#endif

#define _GET_RZ_MATRIX_               \
    CTYPE diagonal_matrix[2];         \
    double c, s;                      \
    c = cos(angle / 2);               \
    s = sin(angle / 2);               \
    diagonal_matrix[0] = c + 1.i * s; \
    diagonal_matrix[1] = c - 1.i * s;

void RZ_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim) {
    _GET_RZ_MATRIX_
    single_qubit_diagonal_matrix_gate(
        target_qubit_index, diagonal_matrix, state, dim);
}

#ifdef _USE_MPI
void RZ_gate_mpi(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim,
    UINT inner_qc) {
    _GET_RZ_MATRIX_
    if (target_qubit_index < inner_qc) {
        single_qubit_diagonal_matrix_gate(
            target_qubit_index, diagonal_matrix, state, dim);
    } else {
        single_qubit_diagonal_matrix_gate_mpi(
            target_qubit_index, diagonal_matrix, state, dim, inner_qc);
    }
}
#endif
