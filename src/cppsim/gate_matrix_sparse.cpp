
#ifndef _MSC_VER
extern "C" {
#include <csim/utility.h>
}
#else
#include <csim/utility.h>
#endif
#include <csim/update_ops_cpp.hpp>

#include "state.hpp"
#include "gate_matrix_sparse.hpp"
#include "type.hpp"
#include <numeric>
#include <algorithm>
#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif

// In construction, "copy" a given matrix. If a given matrix is large, use "move" constructor.
QuantumGateSparseMatrix::QuantumGateSparseMatrix(const std::vector<UINT>& target_qubit_index_list_, const SparseComplexMatrix& matrix_element, const std::vector<UINT>& control_qubit_index_list_) {
	for (auto val : target_qubit_index_list_) {
		this->_target_qubit_list.push_back(TargetQubitInfo(val, 0));
	}
	for (auto val : control_qubit_index_list_) {
		this->_control_qubit_list.push_back(ControlQubitInfo(val, 1));
	}
	this->_matrix_element = SparseComplexMatrix(matrix_element);
	this->_name = "SparseMatrix";
}
QuantumGateSparseMatrix::QuantumGateSparseMatrix(const std::vector<TargetQubitInfo>& target_qubit_index_list_, const SparseComplexMatrix& matrix_element, const std::vector<ControlQubitInfo>& control_qubit_index_list_) {
	this->_target_qubit_list = std::vector<TargetQubitInfo>(target_qubit_index_list_);
	this->_control_qubit_list = std::vector<ControlQubitInfo>(control_qubit_index_list_);
	this->_matrix_element = SparseComplexMatrix(matrix_element);
	this->_name = "SparseMatrix";
}

// In construction, "move" a given matrix, which surpess the cost of copying large matrix element.
QuantumGateSparseMatrix::QuantumGateSparseMatrix(const std::vector<UINT>& target_qubit_index_list_, SparseComplexMatrix* matrix_element, const std::vector<UINT>& control_qubit_index_list_) {
	for (auto val : target_qubit_index_list_) {
		this->_target_qubit_list.push_back(TargetQubitInfo(val, 0));
	}
	for (auto val : control_qubit_index_list_) {
		this->_control_qubit_list.push_back(ControlQubitInfo(val, 1));
	}
	this->_matrix_element.swap(*matrix_element);
	this->_name = "SparseMatrix";
}
QuantumGateSparseMatrix::QuantumGateSparseMatrix(const std::vector<TargetQubitInfo>& target_qubit_index_list_, SparseComplexMatrix* matrix_element, const std::vector<ControlQubitInfo>& control_qubit_index_list_) {
	this->_target_qubit_list = std::vector<TargetQubitInfo>(target_qubit_index_list_);
	this->_control_qubit_list = std::vector<ControlQubitInfo>(control_qubit_index_list_);
	this->_matrix_element.swap(*matrix_element);
	this->_name = "SparseMatrix";
}



void QuantumGateSparseMatrix::update_quantum_state(QuantumStateBase* state) {
	ITYPE dim = 1ULL << state->qubit_count;

	if (this->_control_qubit_list.size() > 0) {
		std::cerr << "Control qubit in sparse matrix gate is not supported" << std::endl;
	}

	std::vector<UINT> target_index;
	std::vector<UINT> control_index;
	std::vector<UINT> control_value;
	for (auto val : this->_target_qubit_list) {
		target_index.push_back(val.index());
	}

	if (state->is_state_vector()) {
#ifdef _USE_GPU
		if (state->get_device_name() == "gpu") {
			std::cerr << "Sparse matrix gate is not supported on GPU" << std::endl;
		}
		else {
			multi_qubit_sparse_matrix_gate_eigen(
				target_index.data(), (UINT)(target_index.size()),
				this->_matrix_element, state->data_c(), dim);
		}
#else
		multi_qubit_sparse_matrix_gate_eigen(
			target_index.data(), (UINT)(target_index.size()),
			this->_matrix_element, state->data_c(), dim);
#endif
	}
	else {
		std::cerr << "not implemented" << std::endl;
	}

}


void QuantumGateSparseMatrix::add_control_qubit(UINT qubit_index, UINT control_value) {
	this->_control_qubit_list.push_back(ControlQubitInfo(qubit_index, control_value));
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

std::ostream& operator<<(std::ostream& os, const QuantumGateSparseMatrix& gate) {
	os << gate.to_string();
	return os;
}
std::ostream& operator<<(std::ostream& os, QuantumGateSparseMatrix* gate) {
	os << *gate;
	return os;
}
