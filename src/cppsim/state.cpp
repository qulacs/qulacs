#include "state.hpp"

#include <csim/stat_ops.hpp>
#include <iostream>

#include "cppsim/gate_matrix.hpp"

namespace state {
CPPCTYPE inner_product(const QuantumState* state1, const QuantumState* state2) {
    if (state1->qubit_count != state2->qubit_count) {
        throw InvalidQubitCountException(
            "Error: inner_product(const QuantumState*, const "
            "QuantumState*): invalid qubit count");
    }

    return state_inner_product(state1->data_c(), state2->data_c(), state1->dim);
}
QuantumState* tensor_product(
    const QuantumState* state_left, const QuantumState* state_right) {
    UINT qubit_count = state_left->qubit_count + state_right->qubit_count;
    QuantumState* qs = new QuantumState(qubit_count);
    state_tensor_product(state_left->data_c(), state_left->dim,
        state_right->data_c(), state_right->dim, qs->data_c());
    return qs;
}
QuantumState* permutate_qubit(
    const QuantumState* state, std::vector<UINT> qubit_order) {
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

}  // namespace state
