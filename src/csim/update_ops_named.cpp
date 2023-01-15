
#include "update_ops.hpp"
#include "utility.hpp"

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void S_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    single_qubit_phase_gate(target_qubit_index, 1.i, state, dim);
}
void Sdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    single_qubit_phase_gate(target_qubit_index, -1.i, state, dim);
}
void T_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    single_qubit_phase_gate(
        target_qubit_index, (1. + 1.i) / sqrt(2.), state, dim);
}
void Tdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    single_qubit_phase_gate(
        target_qubit_index, (1. - 1.i) / sqrt(2.), state, dim);
}
void sqrtX_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    single_qubit_dense_matrix_gate(
        target_qubit_index, SQRT_X_GATE_MATRIX, state, dim);
}
void sqrtXdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    single_qubit_dense_matrix_gate(
        target_qubit_index, SQRT_X_DAG_GATE_MATRIX, state, dim);
}
void sqrtY_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    single_qubit_dense_matrix_gate(
        target_qubit_index, SQRT_Y_GATE_MATRIX, state, dim);
}
void sqrtYdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    single_qubit_dense_matrix_gate(
        target_qubit_index, SQRT_Y_DAG_GATE_MATRIX, state, dim);
}
#ifdef _USE_MPI
void S_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    single_qubit_phase_gate_mpi(target_qubit_index, 1.i, state, dim, inner_qc);
}
void Sdag_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    single_qubit_phase_gate_mpi(target_qubit_index, -1.i, state, dim, inner_qc);
}
void T_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    single_qubit_phase_gate_mpi(
        target_qubit_index, (1. + 1.i) / sqrt(2.), state, dim, inner_qc);
}
void Tdag_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    single_qubit_phase_gate_mpi(
        target_qubit_index, (1. - 1.i) / sqrt(2.), state, dim, inner_qc);
}
void sqrtX_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    single_qubit_dense_matrix_gate_mpi(
        target_qubit_index, SQRT_X_GATE_MATRIX, state, dim, inner_qc);
}
void sqrtXdag_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    single_qubit_dense_matrix_gate_mpi(
        target_qubit_index, SQRT_X_DAG_GATE_MATRIX, state, dim, inner_qc);
}
void sqrtY_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    single_qubit_dense_matrix_gate_mpi(
        target_qubit_index, SQRT_Y_GATE_MATRIX, state, dim, inner_qc);
}
void sqrtYdag_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    single_qubit_dense_matrix_gate_mpi(
        target_qubit_index, SQRT_Y_DAG_GATE_MATRIX, state, dim, inner_qc);
}
#endif
