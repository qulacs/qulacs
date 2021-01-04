

#include "state.hpp"

#ifndef _MSC_VER
extern "C" {
#include <csim/stat_ops.h>
}
#else
#include <csim/stat_ops.h>
#endif
#include <iostream>

namespace state {
CPPCTYPE inner_product(const StateVector* state1, const StateVector* state2) {
    if (state1->qubit_count != state2->qubit_count) {
        std::cerr << "Error: inner_product(const StateVector*, const "
                     "StateVector*): invalid qubit count"
                  << std::endl;
        return 0.;
    }

    return state_inner_product(state1->data_c(), state2->data_c(), state1->dim);
}
StateVector* tensor_product(
    const StateVector* state_left, const StateVector* state_right) {
    UINT qubit_count = state_left->qubit_count + state_right->qubit_count;
    StateVector* qs = new StateVector(qubit_count);
    state_tensor_product(state_left->data_c(), state_left->dim,
        state_right->data_c(), state_right->dim, qs->data_c());
    return qs;
}
StateVector* permutate_qubit(
    const StateVector* state, std::vector<UINT> qubit_order) {
    if (state->qubit_count != (UINT)qubit_order.size()) {
        std::cerr << "Error: permutate_qubit(const StateVector*, "
                     "std::vector<UINT>): invalid qubit count"
                  << std::endl;
        return NULL;
    }
    UINT qubit_count = state->qubit_count;
    StateVector* qs = new StateVector(qubit_count);
    state_permutate_qubit(qubit_order.data(), state->data_c(), qs->data_c(),
        state->qubit_count, state->dim);
    return qs;
}
StateVector* drop_qubit(const StateVector* state, std::vector<UINT> target,
    std::vector<UINT> projection) {
    if (state->qubit_count <= target.size() ||
        target.size() != projection.size()) {
        std::cerr
            << "Error: drop_qubit(const StateVector*, std::vector<UINT>): "
               "invalid qubit count"
            << std::endl;
        return NULL;
    }
    UINT qubit_count = state->qubit_count - (UINT)target.size();
    StateVector* qs = new StateVector(qubit_count);
    state_drop_qubits(target.data(), projection.data(), (UINT)target.size(),
        state->data_c(), qs->data_c(), state->dim);
    return qs;
}
}  // namespace state
