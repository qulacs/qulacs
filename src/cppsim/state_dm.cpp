

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
        throw InvalidQubitCountException(
            "Error: permutate_qubit(const QuantumState*, "
            "std::vector<UINT>): invalid qubit count");
    }
    UINT qubit_count = state->qubit_count;
    DensityMatrixCpu* qs = new DensityMatrixCpu(qubit_count);
    dm_state_permutate_qubit(qubit_order.data(), state->data_c(), qs->data_c(),
        state->qubit_count, state->dim);
    return qs;
}
DensityMatrixCpu* partial_trace(
    const QuantumStateCpu* state, std::vector<UINT> target_traceout) {
    if (state->qubit_count <= target_traceout.size()) {
        throw InvalidQubitCountException(
            "Error: partial_trace(const QuantumState*, "
            "std::vector<UINT>): invalid qubit count");
    }
    if (state->outer_qc > 0) {
        throw NotImplementedException(
            "Error: partial_trace(const QuantumState*, "
            "std::vector<UINT>) using multi-cpu is not implemented");
    }
    UINT qubit_count = state->qubit_count - (UINT)target_traceout.size();
    DensityMatrixCpu* qs = new DensityMatrixCpu(qubit_count);
    dm_state_partial_trace_from_state_vector(target_traceout.data(),
        (UINT)(target_traceout.size()), state->data_c(), qs->data_c(),
        state->dim);
    return qs;
}
DensityMatrixCpu* partial_trace(
    const DensityMatrixCpu* state, std::vector<UINT> target_traceout) {
    if (state->qubit_count <= target_traceout.size()) {
        throw InvalidQubitCountException(
            "Error: partial_trace(const QuantumState*, "
            "std::vector<UINT>): invalid qubit count");
    }
    UINT qubit_count = state->qubit_count - (UINT)target_traceout.size();
    DensityMatrixCpu* qs = new DensityMatrixCpu(qubit_count);
    dm_state_partial_trace_from_density_matrix(target_traceout.data(),
        (UINT)target_traceout.size(), state->data_c(), qs->data_c(),
        state->dim);
    return qs;
}
DensityMatrixCpu* make_mixture(CPPCTYPE prob1, const QuantumStateBase* state1,
    CPPCTYPE prob2, const QuantumStateBase* state2) {
    if (state1->qubit_count != state2->qubit_count) {
        throw InvalidQubitCountException(
            "Error: make_mixture(CPPCTYPE, const QuantumStateBase*, "
            "CPPCTYPE, const QuantumStateBase*): invalid qubit count");
    }
    if ((state1->outer_qc > 0) || (state2->outer_qc > 0)) {
        throw NotImplementedException(
            "Error: make_mixture(CPPCTYPE, const QuantumStateBase*, "
            "CPPCTYPE, const QuantumStateBase*): invalid qubit count "
            "using multi-cpu is not implemented");
    }
    UINT qubit_count = state1->qubit_count;
    DensityMatrixCpu* dm1 = new DensityMatrixCpu(qubit_count);
    dm1->load(state1);
    DensityMatrixCpu* dm2 = new DensityMatrixCpu(qubit_count);
    dm2->load(state2);
    DensityMatrixCpu* mixture = new DensityMatrixCpu(qubit_count);
    mixture->set_zero_norm_state();
    mixture->add_state_with_coef(prob1, dm1);
    mixture->add_state_with_coef(prob2, dm2);
    delete dm1;
    delete dm2;
    return mixture;
}
DllExport QuantumStateBase* from_ptree(const boost::property_tree::ptree& pt) {
    std::string name = pt.get<std::string>("name");
    if (name == "QuantumState") {
        UINT qubit_count = pt.get<UINT>("qubit_count");
        std::vector<UINT> classical_register =
            ptree::uint_array_from_ptree(pt.get_child("classical_register"));
        std::vector<CPPCTYPE> state_vector =
            ptree::complex_array_from_ptree(pt.get_child("state_vector"));
        QuantumState* qs = new QuantumState(qubit_count);
        for (UINT i = 0; i < classical_register.size(); i++) {
            qs->set_classical_value(i, classical_register[i]);
        }
        qs->load(state_vector);
        return qs;
    } else if (name == "DensityMatrix") {
        UINT qubit_count = pt.get<UINT>("qubit_count");
        std::vector<UINT> classical_register =
            ptree::uint_array_from_ptree(pt.get_child("classical_register"));
        std::vector<CPPCTYPE> density_matrix =
            ptree::complex_array_from_ptree(pt.get_child("density_matrix"));
        DensityMatrix* dm = new DensityMatrix(qubit_count);
        for (UINT i = 0; i < classical_register.size(); i++) {
            dm->set_classical_value(i, classical_register[i]);
        }
        dm->load(density_matrix);
        return dm;
    }
    throw UnknownPTreePropertyValueException(
        "unknown value for property \"name\":" + name);
}
}  // namespace state
