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

    return state_inner_product(
        state_bra->data_c(), state_ket->data_c(), state_bra->dim);
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

QuantumStateBase* from_json(const std::string& json) {
    std::map<std::string, std::string> attributes =
        json::object_from_json(json);
    auto name_it = attributes.find("name");
    if (name_it == attributes.end()) {
        throw InvalidJSONFormatException(
            "QuantumStateBase is expected, but an attribute named \'name\' is "
            "not found");
    }
    std::string name = json::string_from_json(name_it->second);
    if (name == "QuantumState") {
        auto qubit_count_it = attributes.find("qubit_count");
        if (qubit_count_it == attributes.end()) {
            throw InvalidJSONFormatException(
                "QuantumStateBase is expected, but an attribute named "
                "\'qubit_count\' is "
                "not found");
        }
        auto classical_register_it = attributes.find("classical_register");
        if (classical_register_it == attributes.end()) {
            throw InvalidJSONFormatException(
                "QuantumStateBase is expected, but an attribute named "
                "\'classical_register\' is "
                "not found");
        }
        auto state_vector_it = attributes.find("state_vector");
        if (state_vector_it == attributes.end()) {
            throw InvalidJSONFormatException(
                "QuantumStateBase is expected, but an attribute named "
                "\'state_vector\' is "
                "not found");
        }

        UINT qubit_count = json::uint_from_json(qubit_count_it->second);
        std::vector<ITYPE> classical_register_itype =
            json::uint_array_from_json(classical_register_it->second);
        std::vector<CPPCTYPE> state_vector =
            json::complex_array_from_json(state_vector_it->second);

        QuantumState* res = new QuantumState(qubit_count);
        for (UINT i = 0; i < classical_register_itype.size(); i++) {
            res->set_classical_value(i, classical_register_itype[i]);
        }
        res->load(state_vector);
        return res;
    } else {
        throw InvalidJSONFormatException(
            "name=\"" + name + "\" for QuantumStateBase is unknown");
    }
}
}  // namespace state
