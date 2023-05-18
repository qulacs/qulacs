#ifdef _USE_GPU

#include "state_gpu.hpp"

#include <assert.h>

QuantumStateGpu::QuantumStateGpu(UINT qubit_count_)
    : QuantumStateBase(qubit_count_, true, 0) {
    set_device(0);
    this->_cuda_stream = allocate_cuda_stream_host(1, 0);
    this->_state_vector =
        reinterpret_cast<void*>(allocate_quantum_state_host(this->_dim, 0));
    initialize_quantum_state_host(
        this->data(), _dim, _cuda_stream, device_number);
}

QuantumStateGpu::QuantumStateGpu(UINT qubit_count_, UINT device_number_)
    : QuantumStateBase(qubit_count_, true, device_number_) {
    int num_device = get_num_device();
    assert(device_number_ < num_device);
    set_device(device_number_);
    this->_cuda_stream = allocate_cuda_stream_host(1, device_number_);
    this->_state_vector = reinterpret_cast<void*>(
        allocate_quantum_state_host(this->_dim, device_number_));
    initialize_quantum_state_host(
        this->data(), _dim, _cuda_stream, device_number_);
}

QuantumStateGpu::~QuantumStateGpu() {
    release_quantum_state_host(this->data(), device_number);
    release_cuda_stream_host(this->_cuda_stream, 1, device_number);
}

void QuantumStateGpu::set_zero_state() {
    initialize_quantum_state_host(
        this->data(), _dim, _cuda_stream, device_number);
}

void QuantumStateGpu::set_zero_norm_state() {
    throw NotImplementedException(
        "set_zero_norm_state for QuantumStateGpu is not implemented "
        "yet");
}

void QuantumStateGpu::set_computational_basis(ITYPE comp_basis) {
    set_computational_basis_host(
        comp_basis, _state_vector, _dim, _cuda_stream, device_number);
}

void QuantumStateGpu::set_Haar_random_state() {
    initialize_Haar_random_state_with_seed_host(
        this->data(), _dim, random.int32(), _cuda_stream, device_number);
}

void QuantumStateGpu::set_Haar_random_state(UINT seed) {
    initialize_Haar_random_state_with_seed_host(
        this->data(), _dim, seed, _cuda_stream, device_number);
}

double QuantumStateGpu::get_zero_probability(UINT target_qubit_index) const {
    return M0_prob_host(
        target_qubit_index, this->data(), _dim, _cuda_stream, device_number);
}

double QuantumStateGpu::get_marginal_probability(
    std::vector<UINT> measured_values) const {
    std::vector<UINT> target_index;
    std::vector<UINT> target_value;
    for (UINT i = 0; i < measured_values.size(); ++i) {
        UINT measured_value = measured_values[i];
        if (measured_value == 0 || measured_value == 1) {
            target_index.push_back(i);
            target_value.push_back(measured_value);
        }
    }
    return marginal_prob_host(target_index.data(), target_value.data(),
        (UINT)target_index.size(), this->data(), _dim, _cuda_stream,
        device_number);
}

double QuantumStateGpu::get_entropy() const {
    return measurement_distribution_entropy_host(
        this->data(), _dim, _cuda_stream, device_number);
}

double QuantumStateGpu::get_squared_norm() const {
    return state_norm_squared_host(
        this->data(), _dim, _cuda_stream, device_number);
}

double QuantumStateGpu::get_squared_norm_single_thread() const {
    return state_norm_squared_host(
        this->data(), _dim, _cuda_stream, device_number);
}

void QuantumStateGpu::normalize(double squared_norm) {
    normalize_host(
        squared_norm, this->data(), _dim, _cuda_stream, device_number);
}

void QuantumStateGpu::normalize_single_thread(double squared_norm) {
    normalize_host(
        squared_norm, this->data(), _dim, _cuda_stream, device_number);
}

QuantumStateBase* QuantumStateGpu::allocate_buffer() const {
    QuantumStateGpu* new_state =
        new QuantumStateGpu(this->_qubit_count, device_number);
    return new_state;
}

QuantumStateGpu* QuantumStateGpu::copy() const {
    QuantumStateGpu* new_state =
        new QuantumStateGpu(this->_qubit_count, device_number);
    copy_quantum_state_from_device_to_device(
        new_state->data(), _state_vector, _dim, _cuda_stream, device_number);
    for (UINT i = 0; i < _classical_register.size(); ++i)
        new_state->set_classical_value(i, _classical_register[i]);
    return new_state;
}

void QuantumStateGpu::load(const QuantumStateBase* _state) {
    if (!_state->is_state_vector()) {
        throw InoperatableQuantumStateTypeException(
            "Error: QuantumStateGpu::load(const QuantumStateBase*): "
            "cannot load DensityMatrix to StateVector");
    }
    if (_state->get_device_name() == "gpu") {
        copy_quantum_state_from_device_to_device(
            this->data(), _state->data(), dim, _cuda_stream, device_number);
    } else {
        this->load(_state->data_cpp());
    }
    this->_classical_register = _state->classical_register;
}

void QuantumStateGpu::load(const std::vector<CPPCTYPE>& _state) {
    copy_quantum_state_from_cppstate_host(
        this->data(), _state.data(), dim, _cuda_stream, device_number);
}

void QuantumStateGpu::load(const CPPCTYPE* _state) {
    copy_quantum_state_from_cppstate_host(
        this->data(), _state, dim, _cuda_stream, device_number);
}

const std::string QuantumStateGpu::get_device_name() const { return "gpu"; }

CPPCTYPE* QuantumStateGpu::data_cpp() const {
    throw NotImplementedException(
        "Cannot reinterpret state vector on GPU to cpp complex "
        "vector. Use duplicate_data_cpp instead.");
}

CTYPE* QuantumStateGpu::data_c() const {
    throw NotImplementedException(
        "Cannot reinterpret state vector on GPU to C complex vector. "
        "Use duplicate_data_cpp instead.");
}

void* QuantumStateGpu::data() const {
    return reinterpret_cast<void*>(this->_state_vector);
}

CTYPE* QuantumStateGpu::duplicate_data_c() const {
    CTYPE* _copy_state = (CTYPE*)malloc(sizeof(CTYPE) * dim);
    get_quantum_state_host(
        this->_state_vector, _copy_state, dim, _cuda_stream, device_number);
    return _copy_state;
}

CPPCTYPE* QuantumStateGpu::duplicate_data_cpp() const {
    CPPCTYPE* _copy_state = (CPPCTYPE*)malloc(sizeof(CPPCTYPE) * dim);
    get_quantum_state_host(
        this->_state_vector, _copy_state, dim, _cuda_stream, device_number);
    return _copy_state;
}

void QuantumStateGpu::add_state(const QuantumStateBase* state) {
    if (!state->is_state_vector()) {
        throw InoperatableQuantumStateTypeException(
            "Error: QuantumStateGpu::add_state(const QuantumStateBase*): "
            "cannot add DensityMatrix to StateVector");
    }
    state_add_host(
        state->data(), this->data(), this->dim, _cuda_stream, device_number);
}

void QuantumStateGpu::add_state_with_coef(
    CPPCTYPE coef, const QuantumStateBase* state) {
    if (!state->is_state_vector()) {
        throw InoperatableQuantumStateTypeException(
            "Error: QuantumStateGpu::add_state_with_coef(CPPCTYPE, "
            "const QuantumStateBase*): "
            "cannot add DensityMatrix to StateVector");
    }
    state_multiply_host(
        coef, this->data(), this->dim, _cuda_stream, device_number);
    state_add_host(
        state->data(), this->data(), this->dim, _cuda_stream, device_number);
    state_multiply_host(CPPCTYPE(1) / coef, this->data(), this->dim,
        _cuda_stream, device_number);
}

void QuantumStateGpu::add_state_with_coef_single_thread(
    CPPCTYPE coef, const QuantumStateBase* state) {
    if (!state->is_state_vector()) {
        throw InoperatableQuantumStateTypeException(
            "Error: "
            "QuantumStateGpu::add_state_with_coef_single_thread(CPPCTYPE, "
            "const QuantumStateBase*): "
            "cannot add DensityMatrix to StateVector");
    }
    state_multiply_host(CPPCTYPE(1) / coef, this->data(), this->dim,
        _cuda_stream, device_number);
    state_add_host(
        state->data(), this->data(), this->dim, _cuda_stream, device_number);
    state_multiply_host(
        coef, this->data(), this->dim, _cuda_stream, device_number);
}

void QuantumStateGpu::multiply_coef(CPPCTYPE coef) {
    state_multiply_host(
        coef, this->data(), this->dim, _cuda_stream, device_number);
}

void QuantumStateGpu::multiply_elementwise_function(
    const std::function<CPPCTYPE(ITYPE)>& func) {
    std::vector<CPPCTYPE> diagonal_matrix(dim);
    for (ITYPE i = 0; i < dim; ++i) {
        diagonal_matrix[i] = func(i);
    }
    multi_qubit_diagonal_matrix_gate_host(
        diagonal_matrix.data(), this->data(), dim, _cuda_stream, device_number);
}

std::vector<ITYPE> QuantumStateGpu::sampling(UINT sampling_count) {
    std::vector<double> stacked_prob;
    std::vector<ITYPE> result;
    double sum = 0.;
    auto ptr = this->duplicate_data_cpp();
    stacked_prob.push_back(0.);
    for (UINT i = 0; i < this->dim; ++i) {
        sum += norm(ptr[i]);
        stacked_prob.push_back(sum);
    }

    for (UINT count = 0; count < sampling_count; ++count) {
        double r = random.uniform();
        auto ite =
            std::lower_bound(stacked_prob.begin(), stacked_prob.end(), r);
        auto index = std::distance(stacked_prob.begin(), ite) - 1;
        result.push_back(index);
    }
    free(ptr);
    return result;
}

std::vector<ITYPE> QuantumStateGpu::sampling(
    UINT sampling_count, UINT random_seed) {
    random.set_seed(random_seed);
    return this->sampling(sampling_count);
}

std::string QuantumStateGpu::to_string() const {
    std::stringstream os;
    ComplexVector eigen_state(this->dim);
    auto data = this->duplicate_data_cpp();
    for (UINT i = 0; i < this->dim; ++i) eigen_state[i] = data[i];
    os << " *** Quantum State ***" << std::endl;
    os << " * Qubit Count : " << this->qubit_count << std::endl;
    os << " * Dimension   : " << this->dim << std::endl;
    os << " * State vector : \n" << eigen_state << std::endl;
    free(data);
    return os.str();
}

boost::property_tree::ptree QuantumStateGpu::to_ptree() const {
    throw NotImplementedException(
        "to_ptree for QuantumStateGpu is not implemented "
        "yet");
}

namespace state {
CPPCTYPE inner_product(
    const QuantumStateGpu* state_bra, const QuantumStateGpu* state_ket) {
    ITYPE dim = state_ket->dim;
    unsigned int device_number = state_ket->device_number;
    assert(dim == state_bra->dim);
    assert(device_number == state_bra->device_number);
    void* cuda_stream = allocate_cuda_stream_host(1, device_number);
    CPPCTYPE ret = inner_product_host(
        state_bra->data(), state_ket->data(), dim, cuda_stream, device_number);
    release_cuda_stream_host(cuda_stream, 1, device_number);
    return ret;
}
}  // namespace state
#endif  //_USE_GPU
