
#include "gate_matrix_sparse.hpp"

#include <algorithm>
#include <csim/update_ops_cpp.hpp>
#include <csim/utility.hpp>
#include <numeric>

#include "state.hpp"
#include "type.hpp"
#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif

// In construction, "copy" a given matrix. If a given matrix is large, use
// "move" constructor.
QuantumGateSparseMatrix::QuantumGateSparseMatrix(
    const std::vector<UINT>& target_qubit_index_list_,
    const SparseComplexMatrix& matrix_element,
    const std::vector<UINT>& control_qubit_index_list_) {
    for (auto val : target_qubit_index_list_) {
        this->_target_qubit_list.push_back(TargetQubitInfo(val, 0));
    }
    for (auto val : control_qubit_index_list_) {
        this->_control_qubit_list.push_back(ControlQubitInfo(val, 1));
    }
    this->_matrix_element = SparseComplexMatrix(matrix_element);
    this->_name = "SparseMatrix";
}
QuantumGateSparseMatrix::QuantumGateSparseMatrix(
    const std::vector<TargetQubitInfo>& target_qubit_index_list_,
    const SparseComplexMatrix& matrix_element,
    const std::vector<ControlQubitInfo>& control_qubit_index_list_) {
    this->_target_qubit_list =
        std::vector<TargetQubitInfo>(target_qubit_index_list_);
    this->_control_qubit_list =
        std::vector<ControlQubitInfo>(control_qubit_index_list_);
    this->_matrix_element = SparseComplexMatrix(matrix_element);
    this->_name = "SparseMatrix";
}

// In construction, "move" a given matrix, which surpess the cost of copying
// large matrix element.
QuantumGateSparseMatrix::QuantumGateSparseMatrix(
    const std::vector<UINT>& target_qubit_index_list_,
    SparseComplexMatrix* matrix_element,
    const std::vector<UINT>& control_qubit_index_list_) {
    for (auto val : target_qubit_index_list_) {
        this->_target_qubit_list.push_back(TargetQubitInfo(val, 0));
    }
    for (auto val : control_qubit_index_list_) {
        this->_control_qubit_list.push_back(ControlQubitInfo(val, 1));
    }
    this->_matrix_element.swap(*matrix_element);
    this->_name = "SparseMatrix";
}
QuantumGateSparseMatrix::QuantumGateSparseMatrix(
    const std::vector<TargetQubitInfo>& target_qubit_index_list_,
    SparseComplexMatrix* matrix_element,
    const std::vector<ControlQubitInfo>& control_qubit_index_list_) {
    this->_target_qubit_list =
        std::vector<TargetQubitInfo>(target_qubit_index_list_);
    this->_control_qubit_list =
        std::vector<ControlQubitInfo>(control_qubit_index_list_);
    this->_matrix_element.swap(*matrix_element);
    this->_name = "SparseMatrix";
}

void QuantumGateSparseMatrix::update_quantum_state(QuantumStateBase* state) {
    ITYPE dim = 1ULL << state->qubit_count;

    if (this->_control_qubit_list.size() > 0) {
        std::stringstream error_message_stream;
        error_message_stream
            << "Control qubit in sparse matrix gate is not supported";
        throw std::invalid_argument(error_message_stream.str());
    }

    std::vector<UINT> target_index;
    std::transform(this->_target_qubit_list.cbegin(),
        this->_target_qubit_list.cend(), std::back_inserter(target_index),
        [](auto value) { return value.index(); });

    if (state->is_state_vector()) {
#ifdef _USE_GPU
        if (state->get_device_name() == "gpu") {
            std::stringstream error_message_stream;
            error_message_stream
                << "Sparse matrix gate is not supported on GPU";
            throw std::invalid_argument(error_message_stream.str());
        } else {
            multi_qubit_sparse_matrix_gate_eigen(target_index.data(),
                (UINT)(target_index.size()), this->_matrix_element,
                state->data_c(), dim);
        }
#else
        multi_qubit_sparse_matrix_gate_eigen(target_index.data(),
            (UINT)(target_index.size()), this->_matrix_element, state->data_c(),
            dim);
#endif
    } else {
        std::stringstream error_message_stream;
        error_message_stream << "not implemented";
        throw std::invalid_argument(error_message_stream.str());
    }
}

void QuantumGateSparseMatrix::add_control_qubit(
    UINT qubit_index, UINT control_value) {
    this->_control_qubit_list.push_back(
        ControlQubitInfo(qubit_index, control_value));
    this->_gate_property &= (~FLAG_PAULI);
    this->_gate_property &= (~FLAG_GAUSSIAN);
}

std::string QuantumGateSparseMatrix::to_string() const {
    std::stringstream os;
    os << QuantumGateBase::to_string();
    os << " * Matrix" << std::endl;
    os << this->_matrix_element << std::endl;
    return os.str();
}

std::ostream& operator<<(
    std::ostream& os, const QuantumGateSparseMatrix& gate) {
    os << gate.to_string();
    return os;
}
std::ostream& operator<<(std::ostream& os, QuantumGateSparseMatrix* gate) {
    os << *gate;
    return os;
}
