
#include "gate_matrix.hpp"

#include <algorithm>
#include <csim/update_ops.hpp>
#include <csim/update_ops_dm.hpp>
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
QuantumGateMatrix::QuantumGateMatrix(
    const std::vector<UINT>& target_qubit_index_list_,
    const ComplexMatrix& matrix_element,
    const std::vector<UINT>& control_qubit_index_list_) {
    for (auto val : target_qubit_index_list_) {
        this->_target_qubit_list.push_back(TargetQubitInfo(val, 0));
    }
    for (auto val : control_qubit_index_list_) {
        this->_control_qubit_list.push_back(ControlQubitInfo(val, 1));
    }
    this->_matrix_element = ComplexMatrix(matrix_element);
    this->_name = "DenseMatrix";
}
QuantumGateMatrix::QuantumGateMatrix(
    const std::vector<TargetQubitInfo>& target_qubit_index_list_,
    const ComplexMatrix& matrix_element,
    const std::vector<ControlQubitInfo>& control_qubit_index_list_) {
    this->_target_qubit_list =
        std::vector<TargetQubitInfo>(target_qubit_index_list_);
    this->_control_qubit_list =
        std::vector<ControlQubitInfo>(control_qubit_index_list_);
    this->_matrix_element = ComplexMatrix(matrix_element);
    this->_name = "DenseMatrix";
}

// In construction, "move" a given matrix, which surpess the cost of copying
// large matrix element.
QuantumGateMatrix::QuantumGateMatrix(
    const std::vector<UINT>& target_qubit_index_list_,
    ComplexMatrix* matrix_element,
    const std::vector<UINT>& control_qubit_index_list_) {
    for (auto val : target_qubit_index_list_) {
        this->_target_qubit_list.push_back(TargetQubitInfo(val, 0));
    }
    for (auto val : control_qubit_index_list_) {
        this->_control_qubit_list.push_back(ControlQubitInfo(val, 1));
    }
    this->_matrix_element.swap(*matrix_element);
    this->_name = "DenseMatrix";
}
QuantumGateMatrix::QuantumGateMatrix(
    const std::vector<TargetQubitInfo>& target_qubit_index_list_,
    ComplexMatrix* matrix_element,
    const std::vector<ControlQubitInfo>& control_qubit_index_list_) {
    this->_target_qubit_list =
        std::vector<TargetQubitInfo>(target_qubit_index_list_);
    this->_control_qubit_list =
        std::vector<ControlQubitInfo>(control_qubit_index_list_);
    this->_matrix_element.swap(*matrix_element);
    this->_name = "DenseMatrix";
}

void QuantumGateMatrix::update_quantum_state(QuantumStateBase* state) {
    // Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
    // Eigen::RowMajor> row_matrix(this->_matrix_element); const CTYPE*
    // matrix_ptr = reinterpret_cast<const CTYPE*>(row_matrix.data()); const
    // CTYPE* matrix_ptr = reinterpret_cast<const
    // CTYPE*>(this->_matrix_element.data());
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
    const CTYPE* matrix_ptr =
        reinterpret_cast<const CTYPE*>(this->_matrix_element.data());
    // #endif

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

    // dense matrix gate for Dense Matrix type simulation
    if (!state->is_state_vector()) {
        if (this->_control_qubit_list.size() == 0) {
            if (this->_target_qubit_list.size() == 1) {
                dm_single_qubit_dense_matrix_gate(
                    target_index[0], matrix_ptr, state->data_c(), state->dim);
            } else {
                dm_multi_qubit_dense_matrix_gate(target_index.data(),
                    (UINT)target_index.size(), matrix_ptr, state->data_c(),
                    state->dim);
            }
        } else {
            if (this->_target_qubit_list.size() == 1) {
                dm_multi_qubit_control_single_qubit_dense_matrix_gate(
                    control_index.data(), control_value.data(),
                    (UINT)control_index.size(), target_index[0], matrix_ptr,
                    state->data_c(), state->dim);
            } else {
                dm_multi_qubit_control_multi_qubit_dense_matrix_gate(
                    control_index.data(), control_value.data(),
                    (UINT)control_index.size(), target_index.data(),
                    (UINT)target_index.size(), matrix_ptr, state->data_c(),
                    state->dim);
            }
        }
        return;
    }

    // dense matrix gate for State Vector type simulation
    // single qubit dense matrix gate
    if (this->_target_qubit_list.size() == 1) {
        // no control qubit
        if (this->_control_qubit_list.size() == 0) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                single_qubit_dense_matrix_gate_host(target_index[0],
                    (const CPPCTYPE*)matrix_ptr, state->data(), state->dim,
                    state->get_cuda_stream(), state->device_number);
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0) {
                single_qubit_dense_matrix_gate_mpi(target_index[0], matrix_ptr,
                    state->data_c(), state->dim, state->inner_qc);
            } else
#endif
            {
                single_qubit_dense_matrix_gate(
                    target_index[0], matrix_ptr, state->data_c(), state->dim);
            }
        }
        // single control qubit
        else if (this->_control_qubit_list.size() == 1) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                single_qubit_control_single_qubit_dense_matrix_gate_host(
                    control_index[0], control_value[0], target_index[0],
                    (const CPPCTYPE*)matrix_ptr, state->data(), state->dim,
                    state->get_cuda_stream(), state->device_number);
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0) {
                single_qubit_control_single_qubit_dense_matrix_gate_mpi(
                    control_index[0], control_value[0], target_index[0],
                    matrix_ptr, state->data_c(), state->dim, state->inner_qc);
            } else
#endif
            {
                single_qubit_control_single_qubit_dense_matrix_gate(
                    control_index[0], control_value[0], target_index[0],
                    matrix_ptr, state->data_c(), state->dim);
            }
        } else {
            // multiple control qubits
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                //
                // std::cerr << "Redirected to multi_control multi_target in
                // GPU" << std::endl;
                multi_qubit_control_multi_qubit_dense_matrix_gate_host(
                    control_index.data(), control_value.data(),
                    (UINT)(control_index.size()), target_index.data(),
                    (UINT)(target_index.size()), (const CPPCTYPE*)matrix_ptr,
                    state->data(), state->dim, state->get_cuda_stream(),
                    state->device_number);
                // exit(0);
                /*
                multi_qubit_control_single_qubit_dense_matrix_gate_host(
                        control_index.data(), control_value.data(),
                (UINT)(control_index.size()), target_index[0], matrix_ptr,
                state->data(), state->dim );
                        */
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0) {
                multi_qubit_control_single_qubit_dense_matrix_gate_mpi(
                    control_index.data(), control_value.data(),
                    (UINT)(control_index.size()), target_index[0], matrix_ptr,
                    state->data_c(), state->dim, state->inner_qc);
            } else
#endif
            {
                multi_qubit_control_single_qubit_dense_matrix_gate(
                    control_index.data(), control_value.data(),
                    (UINT)(control_index.size()), target_index[0], matrix_ptr,
                    state->data_c(), state->dim);
            }
        }
    } else {
        // multi qubit dense matrix gate
        // no control qubit
        if (this->_control_qubit_list.size() == 0) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                multi_qubit_dense_matrix_gate_host(target_index.data(),
                    (UINT)(target_index.size()), (const CPPCTYPE*)matrix_ptr,
                    state->data(), state->dim, state->get_cuda_stream(),
                    state->device_number);
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0) {
                multi_qubit_dense_matrix_gate_mpi(target_index.data(),
                    (UINT)(target_index.size()), matrix_ptr, state->data_c(),
                    state->dim, state->inner_qc);
            } else
#endif
            {
                multi_qubit_dense_matrix_gate(target_index.data(),
                    (UINT)(target_index.size()), matrix_ptr, state->data_c(),
                    state->dim);
            }
        }
        // single control qubit
        else if (this->_control_qubit_list.size() == 1) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                single_qubit_control_multi_qubit_dense_matrix_gate_host(
                    control_index[0], control_value[0], target_index.data(),
                    (UINT)(target_index.size()), (const CPPCTYPE*)matrix_ptr,
                    state->data(), state->dim, state->get_cuda_stream(),
                    state->device_number);
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0)
                single_qubit_control_multi_qubit_dense_matrix_gate_mpi(
                    control_index[0], control_value[0], target_index.data(),
                    (UINT)(target_index.size()), matrix_ptr, state->data_c(),
                    state->dim, state->inner_qc);
            else
#endif
            {
                single_qubit_control_multi_qubit_dense_matrix_gate(
                    control_index[0], control_value[0], target_index.data(),
                    (UINT)(target_index.size()), matrix_ptr, state->data_c(),
                    state->dim);
            }
        }
        // multiple control qubit
        else {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                multi_qubit_control_multi_qubit_dense_matrix_gate_host(
                    control_index.data(), control_value.data(),
                    (UINT)(control_index.size()), target_index.data(),
                    (UINT)(target_index.size()), (const CPPCTYPE*)matrix_ptr,
                    state->data(), state->dim, state->get_cuda_stream(),
                    state->device_number);
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0)
                multi_qubit_control_multi_qubit_dense_matrix_gate_mpi(
                    control_index.data(), control_value.data(),
                    (UINT)(control_index.size()), target_index.data(),
                    (UINT)(target_index.size()), matrix_ptr, state->data_c(),
                    state->dim, state->inner_qc);
            else
#endif
            {
                multi_qubit_control_multi_qubit_dense_matrix_gate(
                    control_index.data(), control_value.data(),
                    (UINT)(control_index.size()), target_index.data(),
                    (UINT)(target_index.size()), matrix_ptr, state->data_c(),
                    state->dim);
            }
        }
    }
}

void QuantumGateMatrix::add_control_qubit(
    UINT qubit_index, UINT control_value) {
    this->_control_qubit_list.push_back(
        ControlQubitInfo(qubit_index, control_value));
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

boost::property_tree::ptree QuantumGateMatrix::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "DenseMatrixGate");
    pt.put_child("target_qubit_list", ptree::to_ptree(_target_qubit_list));
    pt.put_child("control_qubit_list", ptree::to_ptree(_control_qubit_list));
    pt.put_child("matrix", ptree::to_ptree(_matrix_element));
    return pt;
}

std::ostream& operator<<(std::ostream& os, const QuantumGateMatrix& gate) {
    os << gate.to_string();
    return os;
}
std::ostream& operator<<(std::ostream& os, QuantumGateMatrix* gate) {
    os << *gate;
    return os;
}

QuantumGateMatrix* QuantumGateMatrix::get_inverse(void) const {
    return gate::get_adjoint_gate(this);
}
