#include "state_dm.hpp"

#include <csim/stat_ops_dm.hpp>
#include <iostream>

DensityMatrixCpu::DensityMatrixCpu(UINT qubit_count_)
    : QuantumStateBase(qubit_count_, false) {
    this->_density_matrix =
        reinterpret_cast<CPPCTYPE*>(dm_allocate_quantum_state(this->_dim));
    dm_initialize_quantum_state(this->data_c(), _dim);
}

DensityMatrixCpu::~DensityMatrixCpu() {
    dm_release_quantum_state(this->data_c());
}

void DensityMatrixCpu::set_zero_state() {
    dm_initialize_quantum_state(this->data_c(), _dim);
}

void DensityMatrixCpu::set_zero_norm_state() {
    set_zero_state();
    _density_matrix[0] = 0.;
}

void DensityMatrixCpu::set_computational_basis(ITYPE comp_basis) {
    if (comp_basis >= (ITYPE)(1ULL << this->qubit_count)) {
        throw MatrixIndexOutOfRangeException(
            "Error: DensityMatrixCpu::set_computational_basis(ITYPE): "
            "index "
            "of computational basis must be smaller than 2^qubit_count");
    }
    set_zero_state();
    _density_matrix[0] = 0.;
    _density_matrix[comp_basis * dim + comp_basis] = 1.;
}

void DensityMatrixCpu::set_Haar_random_state() {
    this->set_Haar_random_state(random.int32());
}

void DensityMatrixCpu::set_Haar_random_state(UINT seed) {
    QuantumStateCpu* pure_state = new QuantumStateCpu(qubit_count);
    pure_state->set_Haar_random_state(seed);
    dm_initialize_with_pure_state(this->data_c(), pure_state->data_c(), _dim);
    delete pure_state;
}

double DensityMatrixCpu::get_zero_probability(UINT target_qubit_index) const {
    if (target_qubit_index >= this->qubit_count) {
        throw QubitIndexOutOfRangeException(
            "Error: DensityMatrixCpu::get_zero_probability(UINT): index "
            "of target qubit must be smaller than qubit_count");
    }
    return dm_M0_prob(target_qubit_index, this->data_c(), _dim);
}

double DensityMatrixCpu::get_marginal_probability(
    std::vector<UINT> measured_values) const {
    if (measured_values.size() != this->qubit_count) {
        throw InvalidQubitCountException(
            "Error: "
            "DensityMatrixCpu::get_marginal_probability(vector<UINT>): "
            "the length of measured_values must be equal to qubit_count");
    }

    std::vector<UINT> target_index;
    std::vector<UINT> target_value;
    for (UINT i = 0; i < measured_values.size(); ++i) {
        UINT measured_value = measured_values[i];
        if (measured_value == 0 || measured_value == 1) {
            target_index.push_back(i);
            target_value.push_back(measured_value);
        }
    }
    return dm_marginal_prob(target_index.data(), target_value.data(),
        (UINT)target_index.size(), this->data_c(), _dim);
}

double DensityMatrixCpu::get_entropy() const {
    return dm_measurement_distribution_entropy(this->data_c(), _dim);
}

double DensityMatrixCpu::get_squared_norm() const {
    return dm_state_norm_squared(this->data_c(), _dim);
}

double DensityMatrixCpu::get_squared_norm_single_thread() const {
    return dm_state_norm_squared(this->data_c(), _dim);
}

void DensityMatrixCpu::normalize(double squared_norm) {
    dm_normalize(squared_norm, this->data_c(), _dim);
}

void DensityMatrixCpu::normalize_single_thread(double squared_norm) {
    dm_normalize(squared_norm, this->data_c(), _dim);
}

DensityMatrixCpu* DensityMatrixCpu::allocate_buffer() const {
    DensityMatrixCpu* new_state = new DensityMatrixCpu(this->_qubit_count);
    return new_state;
}

DensityMatrixCpu* DensityMatrixCpu::copy() const {
    DensityMatrixCpu* new_state = new DensityMatrixCpu(this->_qubit_count);
    memcpy(new_state->data_cpp(), _density_matrix,
        (size_t)(sizeof(CPPCTYPE) * _dim * _dim));
    for (UINT i = 0; i < _classical_register.size(); ++i)
        new_state->set_classical_value(i, _classical_register[i]);
    return new_state;
}

void DensityMatrixCpu::load(const QuantumStateBase* _state) {
    if (_state->qubit_count != this->qubit_count) {
        throw InvalidQubitCountException(
            "Error: DensityMatrixCpu::load(const QuantumStateBase*): "
            "invalid qubit count");
    }
    if (_state->outer_qc > 0) {
        throw NotImplementedException(
            "Error: DensityMatrixCpu::load(const QuantumStateBase*) "
            "using multi-cpu is not implemented");
    }
    if (_state->is_state_vector()) {
        if (_state->get_device_name() == "gpu") {
            auto ptr = _state->duplicate_data_c();
            dm_initialize_with_pure_state(this->data_c(), ptr, dim);
            free(ptr);
        } else {
            dm_initialize_with_pure_state(
                this->data_c(), _state->data_c(), dim);
        }
    } else {
        memcpy(this->data_cpp(), _state->data_cpp(),
            (size_t)(sizeof(CPPCTYPE) * _dim * _dim));
    }
    this->_classical_register = _state->classical_register;
}

void DensityMatrixCpu::load(const std::vector<CPPCTYPE>& _state) {
    if (_state.size() != _dim && _state.size() != _dim * _dim) {
        throw InvalidStateVectorSizeException(
            "Error: DensityMatrixCpu::load(vector<Complex>&): invalid "
            "length of state");
    }
    if (_state.size() == _dim) {
        dm_initialize_with_pure_state(
            this->data_c(), (const CTYPE*)_state.data(), dim);
    } else {
        memcpy(this->data_cpp(), _state.data(),
            (size_t)(sizeof(CPPCTYPE) * _dim * _dim));
    }
}

void DensityMatrixCpu::load(const Eigen::VectorXcd& _state) {
    ITYPE arg_dim = _state.size();
    if (arg_dim != _dim && arg_dim != _dim * _dim) {
        throw InvalidStateVectorSizeException(
            "Error: DensityMatrixCpu::load(vector<Complex>&): invalid "
            "length of state");
    }
    if (arg_dim == _dim) {
        dm_initialize_with_pure_state(
            this->data_c(), (const CTYPE*)_state.data(), dim);
    } else {
        memcpy(this->data_cpp(), _state.data(),
            (size_t)(sizeof(CPPCTYPE) * _dim * _dim));
    }
}

void DensityMatrixCpu::load(const ComplexMatrix& _state) {
    ITYPE arg_cols = _state.cols();
    ITYPE arg_rows = _state.rows();
    if (arg_cols != _dim && arg_rows != _dim * _dim) {
        throw InvalidStateVectorSizeException(
            "Error: DensityMatrixCpu::load(ComplexMatrix&): invalid "
            "length of state");
    }
    memcpy(this->data_cpp(), _state.data(),
        (size_t)(sizeof(CPPCTYPE) * _dim * _dim));
}

void DensityMatrixCpu::load(const CPPCTYPE* _state) {
    memcpy(this->data_cpp(), _state, (size_t)(sizeof(CPPCTYPE) * _dim * _dim));
}

const std::string DensityMatrixCpu::get_device_name() const { return "cpu"; }

void* DensityMatrixCpu::data() const {
    return reinterpret_cast<void*>(this->_density_matrix);
}

CPPCTYPE* DensityMatrixCpu::data_cpp() const { return this->_density_matrix; }

CTYPE* DensityMatrixCpu::data_c() const {
    return reinterpret_cast<CTYPE*>(this->_density_matrix);
}

CTYPE* DensityMatrixCpu::duplicate_data_c() const {
    CTYPE* new_data = (CTYPE*)malloc(sizeof(CTYPE) * _dim * _dim);
    memcpy(new_data, this->data(), (size_t)(sizeof(CTYPE) * _dim * _dim));
    return new_data;
}

CPPCTYPE* DensityMatrixCpu::duplicate_data_cpp() const {
    CPPCTYPE* new_data = (CPPCTYPE*)malloc(sizeof(CPPCTYPE) * _dim * _dim);
    memcpy(new_data, this->data(), (size_t)(sizeof(CPPCTYPE) * _dim * _dim));
    return new_data;
}

void DensityMatrixCpu::add_state(const QuantumStateBase* state) {
    if (state->is_state_vector()) {
        throw NotImplementedException(
            "add state between density matrix and state vector is not "
            "implemented");
    }
    dm_state_add(state->data_c(), this->data_c(), this->dim);
}

void DensityMatrixCpu::add_state_with_coef(
    CPPCTYPE coef, const QuantumStateBase* state) {
    if (state->is_state_vector()) {
        throw NotImplementedException(
            "add state between density matrix and state vector is not "
            "implemented");
    }
    dm_state_add_with_coef(coef, state->data_c(), this->data_c(), this->dim);
}

void DensityMatrixCpu::add_state_with_coef_single_thread(
    CPPCTYPE coef, const QuantumStateBase* state) {
    if (state->is_state_vector()) {
        throw NotImplementedException(
            "add state between density matrix and state vector is not "
            "implemented");
    }
    dm_state_add_with_coef(coef, state->data_c(), this->data_c(), this->dim);
}

void DensityMatrixCpu::multiply_coef(CPPCTYPE coef) {
    dm_state_multiply(coef, this->data_c(), this->dim);
}

void DensityMatrixCpu::multiply_elementwise_function(
    const std::function<CPPCTYPE(ITYPE)>&) {
    throw NotImplementedException(
        "multiply_elementwise_function for density matrix is not "
        "implemented");
}

std::vector<ITYPE> DensityMatrixCpu::sampling(UINT sampling_count) {
    std::vector<double> stacked_prob;
    std::vector<ITYPE> result;
    double sum = 0.;
    auto ptr = this->data_cpp();
    stacked_prob.push_back(0.);
    for (UINT i = 0; i < this->dim; ++i) {
        sum += abs(ptr[i * dim + i]);
        stacked_prob.push_back(sum);
    }

    for (UINT count = 0; count < sampling_count; ++count) {
        double r = random.uniform();
        auto ite =
            std::lower_bound(stacked_prob.begin(), stacked_prob.end(), r);
        auto index = std::distance(stacked_prob.begin(), ite) - 1;
        result.push_back(index);
    }
    return result;
}

std::vector<ITYPE> DensityMatrixCpu::sampling(
    UINT sampling_count, UINT random_seed) {
    random.set_seed(random_seed);
    return this->sampling(sampling_count);
}

std::string DensityMatrixCpu::to_string() const {
    std::stringstream os;
    ComplexMatrix eigen_state(this->dim, this->dim);
    auto data = this->data_cpp();
    for (UINT i = 0; i < this->dim; ++i) {
        for (UINT j = 0; j < this->dim; ++j) {
            eigen_state(i, j) = data[i * dim + j];
        }
    }
    os << " *** Density Matrix ***" << std::endl;
    os << " * Qubit Count : " << this->qubit_count << std::endl;
    os << " * Dimension   : " << this->dim << std::endl;
    os << " * Density matrix : \n" << eigen_state << std::endl;
    return os.str();
}

boost::property_tree::ptree DensityMatrixCpu::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "DensityMatrix");
    pt.put("qubit_count", _qubit_count);
    pt.put_child("classical_register", ptree::to_ptree(_classical_register));
    pt.put_child(
        "density_matrix", ptree::to_ptree(std::vector<CPPCTYPE>(
                              _density_matrix, _density_matrix + _dim * _dim)));
    return pt;
}

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
