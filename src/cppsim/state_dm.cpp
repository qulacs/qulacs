

#include "state_dm.hpp"

#include <csim/stat_ops_dm.hpp>
#include <iostream>

namespace state {
DensityMatrixCpu* tensor_product(
    const DensityMatrixCpu* state_left, const DensityMatrixCpu* state_right) {
    UINT qubit_count = state_left->qubit_count + state_right->qubit_count;
    DensityMatrixCpu* qs = new DensityMatrixCpu(qubit_count);
    dm_state_tensor_product(state_left->data_c(), state_left->dim,
        state_right->data_c(), state_right->dim, qs->data_c());
    return qs;
}
DensityMatrixCpu* permutate_qubit(
    const DensityMatrixCpu* state, std::vector<UINT> qubit_order) {
    if (state->qubit_count != (UINT)qubit_order.size()) {
        std::stringstream ss;
        ss << "Error: permutate_qubit(const QuantumState*, "
              "std::vector<UINT>): invalid qubit count";
        throw std::invalid_argument(ss.str());
    }
    UINT qubit_count = state->qubit_count;
    DensityMatrixCpu* qs = new DensityMatrixCpu(qubit_count);
    dm_state_permutate_qubit(qubit_order.data(), state->data_c(), qs->data_c(),
        state->qubit_count, state->dim);
    return qs;
}
DensityMatrixCpu* partial_trace(
    const QuantumStateCpu* state, std::vector<UINT> target) {
    if (state->qubit_count <= target.size()) {
        std::stringstream ss;
        ss << "Error: drop_qubit(const QuantumState*, "
              "std::vector<UINT>): invalid qubit count";
        throw std::invalid_argument(ss.str());
    }
    UINT qubit_count = state->qubit_count - (UINT)target.size();
    DensityMatrixCpu* qs = new DensityMatrixCpu(qubit_count);
    dm_state_partial_trace_from_state_vector(target.data(),
        (UINT)(target.size()), state->data_c(), qs->data_c(), state->dim);
    return qs;
}
DensityMatrixCpu* partial_trace(
    const DensityMatrixCpu* state, std::vector<UINT> target) {
    if (state->qubit_count <= target.size()) {
        std::stringstream ss;
        ss << "Error: drop_qubit(const QuantumState*, "
              "std::vector<UINT>): invalid qubit count";
        throw std::invalid_argument(ss.str());
    }
    UINT qubit_count = state->qubit_count - (UINT)target.size();
    DensityMatrixCpu* qs = new DensityMatrixCpu(qubit_count);
    dm_state_partial_trace_from_density_matrix(target.data(),
        (UINT)target.size(), state->data_c(), qs->data_c(), state->dim);
    return qs;
}
}  // namespace state
