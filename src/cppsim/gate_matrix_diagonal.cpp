
#include "gate_matrix_diagonal.hpp"

#include <algorithm>
#include <csim/update_ops_cpp.hpp>
#include <csim/utility.hpp>
#include <numeric>

#include "gate_merge.hpp"
#include "state.hpp"
#include "type.hpp"
#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif

// In construction, "copy" a given matrix. If a given matrix is large, use
// "move" constructor.
QuantumGateDiagonalMatrix::QuantumGateDiagonalMatrix(
    const std::vector<UINT>& target_qubit_index_list_,
    const ComplexVector& diagonal_element,
    const std::vector<UINT>& control_qubit_index_list_) {
    for (auto val : target_qubit_index_list_) {
        this->_target_qubit_list.push_back(TargetQubitInfo(val, 0));
    }
    for (auto val : control_qubit_index_list_) {
        this->_control_qubit_list.push_back(ControlQubitInfo(val, 1));
    }
    this->_diagonal_element = ComplexVector(diagonal_element);
    this->_name = "DiagonalMatrix";
}
QuantumGateDiagonalMatrix::QuantumGateDiagonalMatrix(
    const std::vector<TargetQubitInfo>& target_qubit_index_list_,
    const ComplexVector& diagonal_element,
    const std::vector<ControlQubitInfo>& control_qubit_index_list_) {
    this->_target_qubit_list =
        std::vector<TargetQubitInfo>(target_qubit_index_list_);
    this->_control_qubit_list =
        std::vector<ControlQubitInfo>(control_qubit_index_list_);
    this->_diagonal_element = ComplexVector(diagonal_element);
    this->_name = "DiagonalMatrix";
}

// In construction, "move" a given matrix, which surpess the cost of copying
// large matrix element.
QuantumGateDiagonalMatrix::QuantumGateDiagonalMatrix(
    const std::vector<UINT>& target_qubit_index_list_,
    ComplexVector* diagonal_element,
    const std::vector<UINT>& control_qubit_index_list_) {
    for (auto val : target_qubit_index_list_) {
        this->_target_qubit_list.push_back(TargetQubitInfo(val, 0));
    }
    for (auto val : control_qubit_index_list_) {
        this->_control_qubit_list.push_back(ControlQubitInfo(val, 1));
    }
    this->_diagonal_element.swap(*diagonal_element);
    this->_name = "DiagonalMatrix";
}
QuantumGateDiagonalMatrix::QuantumGateDiagonalMatrix(
    const std::vector<TargetQubitInfo>& target_qubit_index_list_,
    ComplexVector* diagonal_element,
    const std::vector<ControlQubitInfo>& control_qubit_index_list_) {
    this->_target_qubit_list =
        std::vector<TargetQubitInfo>(target_qubit_index_list_);
    this->_control_qubit_list =
        std::vector<ControlQubitInfo>(control_qubit_index_list_);
    this->_diagonal_element.swap(*diagonal_element);
    this->_name = "DiagonalMatrix";
}

void QuantumGateDiagonalMatrix::update_quantum_state(QuantumStateBase* state) {
    ITYPE dim = 1ULL << state->qubit_count;

    const CTYPE* diagonal_ptr =
        reinterpret_cast<const CTYPE*>(this->_diagonal_element.data());
    // convert list of QubitInfo to list of UINT
    std::vector<UINT> target_index;
    std::vector<UINT> control_index;
    std::vector<UINT> control_value;
    std::transform(this->_target_qubit_list.cbegin(),
        this->_target_qubit_list.cend(), std::back_inserter(target_index),
        [](const TargetQubitInfo& value) { return value.index(); });
    for (auto val : this->_control_qubit_list) {
        control_index.push_back(val.index());
        control_value.push_back(val.control_value());
    }

    // diagonal matrix gate for Dense Matrix type simulation
    if (!state->is_state_vector()) {
        throw NotImplementedException(
            "QuantumGateDiagonalMatrix::update_quantum_state for density "
            "matrix is not implemented");
        return;
    }

    // diagonal matrix gate for State Vector type simulation
    // no control qubit
    if (control_index.size() == 0) {
        // single target qubit
        if (target_index.size() == 1) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                throw NotImplementedException(
                    "Diagonal matrix gate is not supported on GPU");
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0) {
                single_qubit_diagonal_matrix_gate_mpi(target_index[0],
                    diagonal_ptr, state->data_c(), state->dim, state->inner_qc);
            } else
#endif
            {
                single_qubit_diagonal_matrix_gate(
                    target_index[0], diagonal_ptr, state->data_c(), dim);
            }
        } else {
            // multiple target qubits
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                throw NotImplementedException(
                    "Diagonal matrix gate is not supported on GPU");
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0) {
                throw NotImplementedException(
                    "Diagonal matrix gate with multi-target qubits "
                    "is not implemented for MPI");
            } else
#endif
            {
                multi_qubit_diagonal_matrix_gate(target_index.data(),
                    (UINT)(target_index.size()), diagonal_ptr, state->data_c(),
                    dim);
            }
        }
    } else {
        // with control qubit
#ifdef _USE_GPU
        if (state->get_device_name() == "gpu") {
            throw NotImplementedException(
                "Diagonal matrix gate is not supported on GPU");
        } else
#endif
#ifdef _USE_MPI
            if (state->outer_qc > 0) {
            throw NotImplementedException(
                "Diagonal matrix gate with control qubits "
                "is not implemented for MPI");
        } else
#endif
        {
            multi_qubit_control_multi_qubit_diagonal_matrix_gate(
                control_index.data(), control_value.data(),
                (UINT)(control_index.size()), target_index.data(),
                (UINT)(target_index.size()), diagonal_ptr, state->data_c(),
                dim);
        }
    }
}

void QuantumGateDiagonalMatrix::add_control_qubit(
    UINT qubit_index, UINT control_value) {
    this->_control_qubit_list.push_back(
        ControlQubitInfo(qubit_index, control_value));
    this->_gate_property &= (~FLAG_PAULI);
    this->_gate_property &= (~FLAG_GAUSSIAN);
}

std::string QuantumGateDiagonalMatrix::to_string() const {
    std::stringstream os;
    os << QuantumGateBase::to_string();
    os << " * Diagonal element" << std::endl;
    os << this->_diagonal_element << std::endl;
    return os.str();
}

boost::property_tree::ptree QuantumGateDiagonalMatrix::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "DiagonalMatrixGate");
    pt.put_child("target_qubit_list", ptree::to_ptree(_target_qubit_list));
    pt.put_child("control_qubit_list", ptree::to_ptree(_control_qubit_list));
    pt.put_child("vector", ptree::to_ptree(_diagonal_element));
    return pt;
}

std::ostream& operator<<(
    std::ostream& os, const QuantumGateDiagonalMatrix& gate) {
    os << gate.to_string();
    return os;
}
std::ostream& operator<<(std::ostream& os, QuantumGateDiagonalMatrix* gate) {
    os << *gate;
    return os;
}