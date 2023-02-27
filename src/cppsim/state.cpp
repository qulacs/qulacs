#include "state.hpp"

#include <csim/stat_ops.hpp>
#include <iostream>

#include "cppsim/gate_matrix.hpp"

namespace state {
CPPCTYPE inner_product(
    const QuantumState* state_bra, const QuantumState* state_ket) {
    if (state_bra->qubit_count != state_ket->qubit_count) {
        throw InvalidQubitCountException(
            "Error: inner_product(const QuantumState*, const "
            "QuantumState*): invalid qubit count");
    }
    CTYPE result;
#ifdef _USE_MPI
    if ((state_bra->outer_qc == 0) and (state_ket->outer_qc == 0))
#endif
    {
        result = state_inner_product(
            state_bra->data_c(), state_ket->data_c(), state_bra->dim);
    }
#ifdef _USE_MPI
    else {
        result = state_inner_product_mpi(state_bra->data_c(),
            state_ket->data_c(), state_bra->dim, state_ket->dim);
    }
#endif
    return result;
}
QuantumState* tensor_product(
    const QuantumState* state_left, const QuantumState* state_right) {
#ifdef _USE_MPI
    if ((state_left->outer_qc > 0) || (state_right->outer_qc > 0))
        throw NotImplementedException(
            "Error: tensor_product does not support multi-cpu");
#endif
    UINT qubit_count = state_left->qubit_count + state_right->qubit_count;
    QuantumState* qs = new QuantumState(qubit_count);
    state_tensor_product(state_left->data_c(), state_left->dim,
        state_right->data_c(), state_right->dim, qs->data_c());
    return qs;
}
QuantumState* permutate_qubit(
    const QuantumState* state, std::vector<UINT> qubit_order) {
#ifdef _USE_MPI
    if (state->outer_qc > 0)
        throw NotImplementedException(
            "Error: permutate_qubit does not support multi-cpu");
#endif
    if (state->qubit_count != (UINT)qubit_order.size()) {
        throw InvalidQubitCountException(
            "Error: permutate_qubit(const QuantumState*, "
            "std::vector<UINT>): invalid qubit count");
    }
    UINT qubit_count = state->qubit_count;
    QuantumState* qs = new QuantumState(qubit_count);
    state_permutate_qubit(qubit_order.data(), state->data_c(), qs->data_c(),
        state->qubit_count, state->dim);
    return qs;
}
QuantumState* drop_qubit(const QuantumState* state, std::vector<UINT> target,
    std::vector<UINT> projection) {
#ifdef _USE_MPI
    if (state->outer_qc > 0)
        throw NotImplementedException(
            "Error: drop_qubit does not support multi-cpu");
#endif
    if (state->qubit_count <= target.size() ||
        target.size() != projection.size()) {
        throw InvalidQubitCountException(
            "Error: drop_qubit(const QuantumState*, std::vector<UINT>): "
            "invalid qubit count");
    }
    UINT qubit_count = state->qubit_count - (UINT)target.size();
    QuantumState* qs = new QuantumState(qubit_count);
    state_drop_qubits(target.data(), projection.data(), (UINT)target.size(),
        state->data_c(), qs->data_c(), state->dim);
    return qs;
}
QuantumState* make_superposition(CPPCTYPE coef1, const QuantumState* state1,
    CPPCTYPE coef2, const QuantumState* state2) {
    if (state1->qubit_count != state2->qubit_count) {
        throw InvalidQubitCountException(
            "Error: make_superposition(CPPCTYPE, const QuantumState*, "
            "CPPCTYPE, const QuantumState*): invalid qubit count");
    }
    QuantumState* qs = new QuantumState(state1->qubit_count);
    qs->set_zero_norm_state();
    qs->add_state_with_coef(coef1, state1);
    qs->add_state_with_coef(coef2, state2);
    return qs;
}
}  // namespace state
