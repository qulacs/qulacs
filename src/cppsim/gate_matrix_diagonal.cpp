
#ifndef _MSC_VER
extern "C" {
#include <csim/utility.h>
}
#else
#include <csim/utility.h>
#endif
#include <csim/update_ops_cpp.hpp>

#include "state.hpp"
#include "gate_matrix_diagonal.hpp"
#include "type.hpp"
#include <numeric>
#include <algorithm>
#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif

// In construction, "copy" a given matrix. If a given matrix is large, use "move" constructor.
QuantumGateDiagonalMatrix::QuantumGateDiagonalMatrix(const std::vector<UINT>& target_qubit_index_list_, const ComplexVector& diagonal_element, const std::vector<UINT>& control_qubit_index_list_) {
	for (auto val : target_qubit_index_list_) {
		this->_target_qubit_list.push_back(TargetQubitInfo(val, 0));
	}
	for (auto val : control_qubit_index_list_) {
		this->_control_qubit_list.push_back(ControlQubitInfo(val, 1));
	}
	this->_diagonal_element = ComplexVector(diagonal_element);
	this->_name = "DiagonalMatrix";
}
QuantumGateDiagonalMatrix::QuantumGateDiagonalMatrix(const std::vector<TargetQubitInfo>& target_qubit_index_list_, const ComplexVector& diagonal_element, const std::vector<ControlQubitInfo>& control_qubit_index_list_) {
	this->_target_qubit_list = std::vector<TargetQubitInfo>(target_qubit_index_list_);
	this->_control_qubit_list = std::vector<ControlQubitInfo>(control_qubit_index_list_);
	this->_diagonal_element = ComplexVector(diagonal_element);
	this->_name = "DiagonalMatrix";
}

// In construction, "move" a given matrix, which surpess the cost of copying large matrix element.
QuantumGateDiagonalMatrix::QuantumGateDiagonalMatrix(const std::vector<UINT>& target_qubit_index_list_, ComplexVector* diagonal_element, const std::vector<UINT>& control_qubit_index_list_) {
	for (auto val : target_qubit_index_list_) {
		this->_target_qubit_list.push_back(TargetQubitInfo(val, 0));
	}
	for (auto val : control_qubit_index_list_) {
		this->_control_qubit_list.push_back(ControlQubitInfo(val, 1));
	}
	this->_diagonal_element.swap(*diagonal_element);
	this->_name = "DiagonalMatrix";
}
QuantumGateDiagonalMatrix::QuantumGateDiagonalMatrix(const std::vector<TargetQubitInfo>& target_qubit_index_list_, ComplexVector* diagonal_element, const std::vector<ControlQubitInfo>& control_qubit_index_list_) {
	this->_target_qubit_list = std::vector<TargetQubitInfo>(target_qubit_index_list_);
	this->_control_qubit_list = std::vector<ControlQubitInfo>(control_qubit_index_list_);
	this->_diagonal_element.swap(*diagonal_element);
	this->_name = "DiagonalMatrix";
}



void QuantumGateDiagonalMatrix::update_quantum_state(QuantumStateBase* state) {
	ITYPE dim = 1ULL << state->qubit_count;

	if (this->_control_qubit_list.size() > 0) {
		std::cerr << "Control qubit in sparse matrix gate is not supported" << std::endl;
	}


	const CTYPE* diagonal_ptr = reinterpret_cast<const CTYPE*>(this->_diagonal_element.data());
	// convert list of QubitInfo to list of UINT
	std::vector<UINT> target_index;
	std::vector<UINT> control_index;
	std::vector<UINT> control_value;
	for (auto val : this->_target_qubit_list) {
		target_index.push_back(val.index());
	}
	for (auto val : this->_control_qubit_list) {
		control_index.push_back(val.index());
		control_value.push_back(val.control_value());
	}

	if (state->is_state_vector()) {
#ifdef _USE_GPU
		if (state->get_device_name() == "gpu") {
			std::cerr << "Diagonal matrix gate is not supported on GPU" << std::endl;
		}
		else {
			if (control_index.size() == 0) {
				if (target_index.size() == 1) {
					single_qubit_diagonal_matrix_gate(target_index[0], diagonal_ptr, state->data_c(), dim);
				}
				else {
					multi_qubit_diagonal_matrix_gate(
						target_index.data(), (UINT)(target_index.size()), diagonal_ptr, state->data_c(), dim);
				}
			}
			else {
				multi_qubit_control_multi_qubit_diagonal_matrix_gate(
					control_index.data(), control_value.data(), (UINT)(control_index.size()),
					target_index.data(), (UINT)(target_index.size()), diagonal_ptr, state->data_c(), dim);
			}
		}
#else
		if (control_index.size() == 0) {
			if (target_index.size() == 1) {
				single_qubit_diagonal_matrix_gate(target_index[0], diagonal_ptr, state->data_c(), dim);
			}
			else {
				multi_qubit_diagonal_matrix_gate(
					target_index.data(), (UINT)(target_index.size()), diagonal_ptr, state->data_c(), dim);
			}
		}
		else {
			multi_qubit_control_multi_qubit_diagonal_matrix_gate(
				control_index.data(), control_value.data(), (UINT)(control_index.size()),
				target_index.data(), (UINT)(target_index.size()), diagonal_ptr, state->data_c(), dim);
		}
#endif
	}
	else {
		std::cerr << "not implemented" << std::endl;
	}

}


void QuantumGateDiagonalMatrix::add_control_qubit(UINT qubit_index, UINT control_value) {
	this->_control_qubit_list.push_back(ControlQubitInfo(qubit_index, control_value));
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

std::ostream& operator<<(std::ostream& os, const QuantumGateDiagonalMatrix& gate) {
	os << gate.to_string();
	return os;
}
std::ostream& operator<<(std::ostream& os, QuantumGateDiagonalMatrix* gate) {
	os << *gate;
	return os;
}
