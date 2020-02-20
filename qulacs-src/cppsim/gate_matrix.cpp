
#ifndef _MSC_VER
extern "C" {
#include <csim/update_ops.h>
#include <csim/update_ops_dm.h>
#include <csim/utility.h>
}
#else
#include <csim/update_ops.h>
#include <csim/update_ops_dm.h>
#include <csim/utility.h>
#endif

#include "state.hpp"
#include "gate_matrix.hpp"
#include "type.hpp"
#include <numeric>
#include <algorithm>
#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif

// In construction, "copy" a given matrix. If a given matrix is large, use "move" constructor.
QuantumGateMatrix::QuantumGateMatrix(const std::vector<UINT>& target_qubit_index_list_, const ComplexMatrix& matrix_element, const std::vector<UINT>& control_qubit_index_list_){
    for(auto val : target_qubit_index_list_){
        this->_target_qubit_list.push_back(TargetQubitInfo(val,0));
    }
    for(auto val : control_qubit_index_list_){
        this->_control_qubit_list.push_back(ControlQubitInfo(val,1));
    }
    this->_matrix_element = ComplexMatrix(matrix_element);
    this->_name = "DenseMatrix";
}
QuantumGateMatrix::QuantumGateMatrix(const std::vector<TargetQubitInfo>& target_qubit_index_list_, const ComplexMatrix& matrix_element, const std::vector<ControlQubitInfo>& control_qubit_index_list_){
    this->_target_qubit_list = std::vector<TargetQubitInfo>(target_qubit_index_list_);
    this->_control_qubit_list = std::vector<ControlQubitInfo>(control_qubit_index_list_);
    this->_matrix_element = ComplexMatrix(matrix_element);
    this->_name = "DenseMatrix";
}

// In construction, "move" a given matrix, which surpess the cost of copying large matrix element.
QuantumGateMatrix::QuantumGateMatrix(const std::vector<UINT>& target_qubit_index_list_, ComplexMatrix* matrix_element, const std::vector<UINT>& control_qubit_index_list_){
    for(auto val : target_qubit_index_list_){
        this->_target_qubit_list.push_back(TargetQubitInfo(val,0));
    }
    for(auto val : control_qubit_index_list_){
        this->_control_qubit_list.push_back(ControlQubitInfo(val,1));
    }
    this->_matrix_element.swap(*matrix_element);
    this->_name = "DenseMatrix";
}
QuantumGateMatrix::QuantumGateMatrix(const std::vector<TargetQubitInfo>& target_qubit_index_list_, ComplexMatrix* matrix_element, const std::vector<ControlQubitInfo>& control_qubit_index_list_){
    this->_target_qubit_list = std::vector<TargetQubitInfo>(target_qubit_index_list_);
    this->_control_qubit_list = std::vector<ControlQubitInfo>(control_qubit_index_list_);
    this->_matrix_element.swap(*matrix_element);
    this->_name = "DenseMatrix";
}



void QuantumGateMatrix::update_quantum_state(QuantumStateBase* state) {
    ITYPE dim = 1ULL << state->qubit_count;

    //Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_matrix(this->_matrix_element);
    //const CTYPE* matrix_ptr = reinterpret_cast<const CTYPE*>(row_matrix.data());
    //const CTYPE* matrix_ptr = reinterpret_cast<const CTYPE*>(this->_matrix_element.data());
	/*
#ifdef _USE_GPU
	const void* matrix_ptr = NULL;
	if (state->get_device_name() == "gpu") {
        matrix_ptr = reinterpret_cast<const void*>(this->_matrix_element.data());
    }else{
        matrix_ptr = reinterpret_cast<const CTYPE*>(this->_matrix_element.data());
    }
#else
*/
    const CTYPE* matrix_ptr = reinterpret_cast<const CTYPE*>(this->_matrix_element.data());
//#endif

    // convert list of QubitInfo to list of UINT
    std::vector<UINT> target_index;
    std::vector<UINT> control_index;
    std::vector<UINT> control_value;
    for(auto val : this->_target_qubit_list){
        target_index.push_back(val.index());
    }
    for(auto val : this->_control_qubit_list){
        control_index.push_back(val.index());
        control_value.push_back(val.control_value());
    }


	if (state->is_state_vector()) {
		// single qubit dense matrix gate
		if (this->_target_qubit_list.size() == 1) {

			// no control qubit
			if (this->_control_qubit_list.size() == 0) {
#ifdef _USE_GPU
				if (state->get_device_name() == "gpu") {
					single_qubit_dense_matrix_gate_host(
						target_index[0],
						(const CPPCTYPE*)matrix_ptr, state->data(), dim, state->get_cuda_stream(), state->device_number);

				}
				else {
					single_qubit_dense_matrix_gate(
						target_index[0],
						matrix_ptr, state->data_c(), dim);
				}
#else
				single_qubit_dense_matrix_gate(
					target_index[0],
					matrix_ptr, state->data_c(), dim);
#endif
			}
			// single control qubit
			else if (this->_control_qubit_list.size() == 1) {
#ifdef _USE_GPU
				if (state->get_device_name() == "gpu") {
					single_qubit_control_single_qubit_dense_matrix_gate_host(
						control_index[0], control_value[0],
						target_index[0],
						(const CPPCTYPE*)matrix_ptr, state->data(), dim, state->get_cuda_stream(), state->device_number);
				}
				else {
					single_qubit_control_single_qubit_dense_matrix_gate(
						control_index[0], control_value[0],
						target_index[0],
						matrix_ptr, state->data_c(), dim);
				}
#else
				single_qubit_control_single_qubit_dense_matrix_gate(
					control_index[0], control_value[0],
					target_index[0],
					matrix_ptr, state->data_c(), dim);
#endif
			}
			// multiple control qubits
			else {
#ifdef _USE_GPU
				if (state->get_device_name() == "gpu") {
					//
					//std::cerr << "Redirected to multi_control multi_target in GPU" << std::endl;
					multi_qubit_control_multi_qubit_dense_matrix_gate_host(
						control_index.data(), control_value.data(), (UINT)(control_index.size()),
						target_index.data(), (UINT)(target_index.size()),
						(const CPPCTYPE*)matrix_ptr, state->data(), dim, state->get_cuda_stream(), state->device_number);
					//exit(0);
					/*
					multi_qubit_control_single_qubit_dense_matrix_gate_host(
						control_index.data(), control_value.data(), (UINT)(control_index.size()),
						target_index[0],
						matrix_ptr, state->data(), dim );
						*/
				}
				else {
					multi_qubit_control_single_qubit_dense_matrix_gate(
						control_index.data(), control_value.data(), (UINT)(control_index.size()),
						target_index[0],
						matrix_ptr, state->data_c(), dim);
				}
#else
				multi_qubit_control_single_qubit_dense_matrix_gate(
					control_index.data(), control_value.data(), (UINT)(control_index.size()),
					target_index[0],
					matrix_ptr, state->data_c(), dim);
#endif
			}
		}

		// multi qubit dense matrix gate
		else {
			// no control qubit
			if (this->_control_qubit_list.size() == 0) {
#ifdef _USE_GPU
				if (state->get_device_name() == "gpu") {
					multi_qubit_dense_matrix_gate_host(
						target_index.data(), (UINT)(target_index.size()),
						(const CPPCTYPE*)matrix_ptr, state->data(), dim, state->get_cuda_stream(), state->device_number);
				}
				else {
					multi_qubit_dense_matrix_gate(
						target_index.data(), (UINT)(target_index.size()),
						matrix_ptr, state->data_c(), dim);
				}
#else
				multi_qubit_dense_matrix_gate(
					target_index.data(), (UINT)(target_index.size()),
					matrix_ptr, state->data_c(), dim);
#endif
			}
			// single control qubit
			else if (this->_control_qubit_list.size() == 1) {
#ifdef _USE_GPU
				if (state->get_device_name() == "gpu") {
					single_qubit_control_multi_qubit_dense_matrix_gate_host(
						control_index[0], control_value[0],
						target_index.data(), (UINT)(target_index.size()),
						(const CPPCTYPE*)matrix_ptr, state->data(), dim, state->get_cuda_stream(), state->device_number);
				}
				else {
					single_qubit_control_multi_qubit_dense_matrix_gate(
						control_index[0], control_value[0],
						target_index.data(), (UINT)(target_index.size()),
						matrix_ptr, state->data_c(), dim);
				}
#else
				single_qubit_control_multi_qubit_dense_matrix_gate(
					control_index[0], control_value[0],
					target_index.data(), (UINT)(target_index.size()),
					matrix_ptr, state->data_c(), dim);
#endif
			}
			// multiple control qubit
			else {
#ifdef _USE_GPU
				if (state->get_device_name() == "gpu") {
					multi_qubit_control_multi_qubit_dense_matrix_gate_host(
						control_index.data(), control_value.data(), (UINT)(control_index.size()),
						target_index.data(), (UINT)(target_index.size()),
						(const CPPCTYPE*)matrix_ptr, state->data(), dim, state->get_cuda_stream(), state->device_number);
				}
				else {
					multi_qubit_control_multi_qubit_dense_matrix_gate(
						control_index.data(), control_value.data(), (UINT)(control_index.size()),
						target_index.data(), (UINT)(target_index.size()),
						matrix_ptr, state->data_c(), dim);
				}
#else
				multi_qubit_control_multi_qubit_dense_matrix_gate(
					control_index.data(), control_value.data(), (UINT)(control_index.size()),
					target_index.data(), (UINT)(target_index.size()),
					matrix_ptr, state->data_c(), dim);
#endif
			}
		}
	}
	else {
		if (this->_control_qubit_list.size() == 0) {
			if (this->_target_qubit_list.size() == 1) {
				dm_single_qubit_dense_matrix_gate(target_index[0], matrix_ptr, state->data_c(), dim);
			}
			else {
				dm_multi_qubit_dense_matrix_gate(target_index.data(), (UINT)target_index.size(), matrix_ptr, state->data_c(), dim);
			}
		}
		else {
			if (this->_target_qubit_list.size() == 1) {
				dm_multi_qubit_control_single_qubit_dense_matrix_gate(control_index.data(), control_value.data(), (UINT)control_index.size(), target_index[0], matrix_ptr, state->data_c(), dim);
			}
			else {
				dm_multi_qubit_control_multi_qubit_dense_matrix_gate(control_index.data(), control_value.data(), (UINT)control_index.size(), target_index.data(), (UINT)target_index.size(), matrix_ptr, state->data_c(), dim);
			}
		}
	}
}


void QuantumGateMatrix::add_control_qubit(UINT qubit_index, UINT control_value) {
    this->_control_qubit_list.push_back(ControlQubitInfo(qubit_index, control_value));
    this->_gate_property &= (~FLAG_PAULI);
    this->_gate_property &= (~FLAG_GAUSSIAN);
}

std::string QuantumGateMatrix::to_string() const {
    std::stringstream os;
    os << QuantumGateBase::to_string();
    os << " * Matrix" << std::endl;
    os << this->_matrix_element << std::endl;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const QuantumGateMatrix& gate) {
    os << gate.to_string();
    return os;
}
std::ostream& operator<<(std::ostream& os, QuantumGateMatrix* gate) {
    os << *gate;
    return os;
}
