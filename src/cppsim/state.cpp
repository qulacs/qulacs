#include "state.hpp"

#include <csim/stat_ops.hpp>
#include <iostream>

#include "cppsim/gate_matrix.hpp"

QuantumStateBase::QuantumStateBase(UINT qubit_count_, bool is_state_vector)
    : qubit_count(_qubit_count),
      inner_qc(_inner_qc),
      outer_qc(_outer_qc),
      dim(_dim),
      classical_register(_classical_register),
      device_number(_device_number) {
    this->_qubit_count = qubit_count_;
    this->_inner_qc = qubit_count_;
    this->_outer_qc = 0;
    this->_dim = 1ULL << qubit_count_;
    this->_is_state_vector = is_state_vector;
    this->_device_number = 0;
}

QuantumStateBase::QuantumStateBase(
    UINT qubit_count_, bool is_state_vector, int use_multi_cpu)
    : qubit_count(_qubit_count),
      inner_qc(_inner_qc),
      outer_qc(_outer_qc),
      dim(_dim),
      classical_register(_classical_register),
      device_number(_device_number) {
    UINT mpirank;
    UINT mpisize;
    if (use_multi_cpu) {
#ifdef _USE_MPI
        MPIutil& mpiutil = MPIutil::get_inst();
        mpirank = mpiutil.get_rank();
        mpisize = mpiutil.get_size();
#else
        mpirank = 0;
        mpisize = 1;
#endif
    } else {
        mpirank = 0;
        mpisize = 1;
    }
    if ((mpisize & (mpisize - 1))) {
        throw MPISizeException(
            "Error: QuantumStateBase::QuantumStateBase(UINT, bool, bool): "
            "mpi-size must be power of 2");
    }
    UINT log_nodes = std::log2(mpisize);
    if (use_multi_cpu &&
        (qubit_count_ >= (log_nodes + 2))) {  // minimum inner_qc=2
        this->_inner_qc = qubit_count_ - log_nodes;
        this->_outer_qc = log_nodes;
    } else {
        this->_inner_qc = qubit_count_;
        this->_outer_qc = 0;
    }

    this->_qubit_count = qubit_count_;
    this->_dim = 1ULL << this->_inner_qc;
    this->_is_state_vector = is_state_vector;
    this->_device_number = mpirank;
}

QuantumStateBase::QuantumStateBase(
    UINT qubit_count_, bool is_state_vector, UINT device_number_)
    : qubit_count(_qubit_count),
      inner_qc(_inner_qc),
      outer_qc(_outer_qc),
      dim(_dim),
      classical_register(_classical_register),
      device_number(_device_number) {
    this->_qubit_count = qubit_count_;
    this->_inner_qc = qubit_count_;
    this->_outer_qc = 0;
    this->_dim = 1ULL << qubit_count_;
    this->_is_state_vector = is_state_vector;
    this->_device_number = device_number_;
}

QuantumStateBase::~QuantumStateBase() {}

bool QuantumStateBase::is_state_vector() const {
    return this->_is_state_vector;
}

UINT QuantumStateBase::get_classical_value(UINT index) {
    if (_classical_register.size() <= index) {
        _classical_register.resize(index + 1, 0);
    }
    return _classical_register[index];
}

void QuantumStateBase::set_classical_value(UINT index, UINT val) {
    if (_classical_register.size() <= index) {
        _classical_register.resize(index + 1, 0);
    }
    _classical_register[index] = val;
}

const std::vector<UINT> QuantumStateBase::get_classical_register() const {
    return _classical_register;
}

std::string QuantumStateBase::to_string() const {
    std::stringstream os;
    ComplexVector eigen_state(this->dim);
    auto data = this->data_cpp();
    for (UINT i = 0; i < this->dim; ++i) eigen_state[i] = data[i];

    os << " *** Quantum State ***" << std::endl;
    UINT myrank = 0;
#ifdef _USE_MPI
    if (this->outer_qc > 0) {
        MPIutil& mpiutil = MPIutil::get_inst();
        myrank = mpiutil.get_rank();
    }
#endif
    if (myrank == 0) {
        os << " * Qubit Count : " << this->qubit_count << std::endl;
        os << " * Dimension   : " << this->dim << std::endl;
#ifdef _USE_MPI
        os << " * Local Qubit Count : " << this->inner_qc << std::endl;
        os << " * Global Qubit Count : " << this->outer_qc << std::endl;
#endif
    }
    if (this->outer_qc > 0) {
        os << " * Rank : " << myrank << std::endl;
    }
    os << " * State vector : \n" << eigen_state << std::endl;
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const QuantumStateBase& state) {
    os << state.to_string();
    return os;
}

std::ostream& operator<<(std::ostream& os, const QuantumStateBase* state) {
    os << *state;
    return os;
}

void* QuantumStateBase::get_cuda_stream() const { return this->_cuda_stream; }

QuantumStateCpu::QuantumStateCpu(UINT qubit_count_)
    : QuantumStateBase(qubit_count_, true) {
    this->_state_vector =
        reinterpret_cast<CPPCTYPE*>(allocate_quantum_state(this->_dim));
    initialize_quantum_state(this->data_c(), _dim);
}

QuantumStateCpu::QuantumStateCpu(UINT qubit_count_, bool use_multi_cpu)
    : QuantumStateBase(qubit_count_, true, (int)use_multi_cpu) {
    this->_state_vector =
        reinterpret_cast<CPPCTYPE*>(allocate_quantum_state(this->_dim));
#ifdef _USE_MPI
    if (this->outer_qc > 0)
        initialize_quantum_state_mpi(this->data_c(), _dim, this->outer_qc);
    else
#endif
    {
        initialize_quantum_state(this->data_c(), _dim);
    }
}

QuantumStateCpu::~QuantumStateCpu() {
#ifdef _USE_MPI
    if (this->outer_qc > 0) {
        MPIutil& mpiutil = MPIutil::get_inst();
        mpiutil.release_workarea();
    }
#endif
    release_quantum_state(this->data_c());
}

void QuantumStateCpu::set_zero_state() {
#ifdef _USE_MPI
    initialize_quantum_state_mpi(this->data_c(), _dim, this->outer_qc);
#else
    initialize_quantum_state(this->data_c(), _dim);
#endif
}

void QuantumStateCpu::set_zero_norm_state() {
    set_zero_state();
    _state_vector[0] = 0;
}

void QuantumStateCpu::set_computational_basis(ITYPE comp_basis) {
    if (comp_basis >= (ITYPE)(1ULL << this->qubit_count)) {
        throw MatrixIndexOutOfRangeException(
            "Error: QuantumStateCpu::set_computational_basis(ITYPE): "
            "index of "
            "computational basis must be smaller than 2^qubit_count");
    }
    set_zero_state();
    _state_vector[0] = 0.;
#ifdef _USE_MPI
    ITYPE myrank = 0;
    if (this->outer_qc > 0) {
        MPIutil& mpiutil = MPIutil::get_inst();
        myrank = (ITYPE)mpiutil.get_rank();
    }
    if (this->outer_qc == 0 || (comp_basis >> this->inner_qc) == myrank) {
        _state_vector[comp_basis & (this->_dim - 1)] = 1.;
    }
#else
    _state_vector[comp_basis] = 1.;
#endif
}

void QuantumStateCpu::set_Haar_random_state() {
    UINT seed = random.int32();
#ifdef _USE_MPI
    if (this->outer_qc > 0) {
        // すべてのrankで同一の結果を得るために、seedを共有する
        MPIutil& mpiutil = MPIutil::get_inst();
        if (mpiutil.get_size() > 1) mpiutil.s_u_bcast(&seed);
    }
#endif
    set_Haar_random_state(seed);
}

void QuantumStateCpu::set_Haar_random_state(UINT seed) {
#ifdef _USE_MPI
    // 各rankで異なるseedを用いる必要がある
    if (this->outer_qc > 0) {
        MPIutil& mpiutil = MPIutil::get_inst();
        seed += mpiutil.get_rank();
    }
    initialize_Haar_random_state_mpi_with_seed(
        this->data_c(), _dim, this->outer_qc, seed);
#else
    initialize_Haar_random_state_with_seed(this->data_c(), _dim, seed);
#endif
}

double QuantumStateCpu::get_zero_probability(UINT target_qubit_index) const {
#ifdef _USE_MPI
    if (this->outer_qc > 0)
        throw NotImplementedException(
            "Error: get_zero_probability does not support multi-cpu");
#endif
    if (target_qubit_index >= this->qubit_count) {
        throw QubitIndexOutOfRangeException(
            "Error: QuantumStateCpu::get_zero_probability(UINT): index "
            "of target qubit must be smaller than qubit_count");
    }
    return M0_prob(target_qubit_index, this->data_c(), _dim);
}

double QuantumStateCpu::get_marginal_probability(
    std::vector<UINT> measured_values) const {
#ifdef _USE_MPI
    if (this->outer_qc > 0)
        throw NotImplementedException(
            "Error: get_marginal_probability does not support multi-cpu");
#endif
    if (measured_values.size() != this->qubit_count) {
        throw InvalidQubitCountException(
            "Error: "
            "QuantumStateCpu::get_marginal_probability(vector<UINT>): "
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
    return marginal_prob(target_index.data(), target_value.data(),
        (UINT)target_index.size(), this->data_c(), _dim);
}

double QuantumStateCpu::get_entropy() const {
    double entropy = measurement_distribution_entropy(this->data_c(), _dim);
#ifdef _USE_MPI
    MPIutil& mpiutil = MPIutil::get_inst();
    if (this->outer_qc > 0) mpiutil.s_D_allreduce(&entropy);
#endif
    return entropy;
}

double QuantumStateCpu::get_squared_norm() const {
    double norm;
#ifdef _USE_MPI
    if (this->outer_qc > 0) {
        norm = state_norm_squared_mpi(this->data_c(), _dim);
    } else
#endif
    {
        norm = state_norm_squared(this->data_c(), _dim);
    }
    return norm;
}

double QuantumStateCpu::get_squared_norm_single_thread() const {
    return state_norm_squared_single_thread(this->data_c(), _dim);
}

void QuantumStateCpu::normalize(double squared_norm) {
    ::normalize(squared_norm, this->data_c(), _dim);
}

void QuantumStateCpu::normalize_single_thread(double squared_norm) {
    ::normalize_single_thread(squared_norm, this->data_c(), _dim);
}

QuantumStateCpu* QuantumStateCpu::allocate_buffer() const {
    QuantumStateCpu* new_state;
#ifdef _USE_MPI
    if (this->outer_qc > 0)
        new_state = new QuantumStateCpu(this->_qubit_count, true);
    else
        new_state = new QuantumStateCpu(this->_qubit_count, false);
#else
    new_state = new QuantumStateCpu(this->_qubit_count);
#endif
    return new_state;
}

QuantumStateCpu* QuantumStateCpu::copy() const {
    QuantumStateCpu* new_state = this->allocate_buffer();

    memcpy(new_state->data_cpp(), _state_vector,
        (size_t)(sizeof(CPPCTYPE) * _dim));
    for (UINT i = 0; i < _classical_register.size(); ++i) {
        new_state->set_classical_value(i, _classical_register[i]);
    }

    return new_state;
}

void QuantumStateCpu::load(const QuantumStateBase* _state) {
    if (_state->qubit_count != this->qubit_count) {
        throw InvalidQubitCountException(
            "Error: QuantumStateCpu::load(const QuantumStateBase*): "
            "invalid qubit count");
    }
    if (!_state->is_state_vector()) {
        throw InoperatableQuantumStateTypeException(
            "Error: QuantumStateCpu::load(const QuantumStateBase*): "
            "cannot load DensityMatrix to StateVector");
    }

    this->_classical_register = _state->classical_register;
    if (_state->get_device_name() == "gpu") {
        auto ptr = _state->duplicate_data_cpp();
        memcpy(this->data_cpp(), ptr, (size_t)(sizeof(CPPCTYPE) * _dim));
        free(ptr);
#ifdef _USE_MPI
    } else if (_state->outer_qc > 0) {
        MPIutil& mpiutil = MPIutil::get_inst();
        if (this->outer_qc > 0) {
            if (_state->qubit_count != this->qubit_count) {
                throw InvalidQubitCountException(
                    "Error: QuantumStateCpu::load(const QuantumStateBase*)"
                    ": invalid global qubit count");
            }
            // load multicpu to multicpu
            memcpy(this->data_cpp(), _state->data_cpp(),
                (size_t)(sizeof(CPPCTYPE) * _dim));
        } else {
            // load multicpu to cpu
            mpiutil.m_DC_allgather(_state->data_cpp(), this->data_cpp(),
                _dim / mpiutil.get_size());
        }
#endif
    } else {
#ifdef _USE_MPI
        if (this->outer_qc > 0) {
            MPIutil& mpiutil = MPIutil::get_inst();
            // load cpu to multicpu
            ITYPE offs = _dim * mpiutil.get_rank();
            memcpy(this->data_cpp(), _state->data_cpp() + offs,
                (size_t)(sizeof(CPPCTYPE) * _dim));
        } else
#endif
        {
            // load cpu to multicpu
            memcpy(this->data_cpp(), _state->data_cpp(),
                (size_t)(sizeof(CPPCTYPE) * _dim));
        }
    }
}

void QuantumStateCpu::load(const std::vector<CPPCTYPE>& _state) {
    if (_state.size() != _dim) {
        throw InvalidStateVectorSizeException(
            "Error: QuantumStateCpu::load(vector<Complex>&): invalid "
            "length of state");
    }
    memcpy(this->data_cpp(), _state.data(), (size_t)(sizeof(CPPCTYPE) * _dim));
}

void QuantumStateCpu::load(const CPPCTYPE* _state) {
    memcpy(this->data_cpp(), _state, (size_t)(sizeof(CPPCTYPE) * _dim));
}

const std::string QuantumStateCpu::get_device_name() const {
#ifdef _USE_MPI
    if (this->outer_qc > 0) {
        return "multi-cpu";
    } else
#endif
    {
        return "cpu";
    }
}

void* QuantumStateCpu::data() const {
    return reinterpret_cast<void*>(this->_state_vector);
}

CPPCTYPE* QuantumStateCpu::data_cpp() const { return this->_state_vector; }

CTYPE* QuantumStateCpu::data_c() const {
    return reinterpret_cast<CTYPE*>(this->_state_vector);
}

CTYPE* QuantumStateCpu::duplicate_data_c() const {
    CTYPE* new_data = (CTYPE*)malloc(sizeof(CTYPE) * _dim);
    memcpy(new_data, this->data(), (size_t)(sizeof(CTYPE) * _dim));
    return new_data;
}

CPPCTYPE* QuantumStateCpu::duplicate_data_cpp() const {
    CPPCTYPE* new_data = (CPPCTYPE*)malloc(sizeof(CPPCTYPE) * _dim);
    memcpy(new_data, this->data(), (size_t)(sizeof(CPPCTYPE) * _dim));
    return new_data;
}

void QuantumStateCpu::add_state(const QuantumStateBase* state) {
    if (!state->is_state_vector()) {
        throw InoperatableQuantumStateTypeException(
            "Error: QuantumStateCpu::add_state(const QuantumStateBase*): "
            "cannot add DensityMatrix to StateVector");
    }
    if (state->get_device_name() == "gpu") {
        throw QuantumStateProcessorException(
            "State vector on GPU cannot be added to that on CPU");
    }
    state_add(state->data_c(), this->data_c(), this->dim);
}

void QuantumStateCpu::add_state_with_coef(
    CPPCTYPE coef, const QuantumStateBase* state) {
    if (!state->is_state_vector()) {
        throw InoperatableQuantumStateTypeException(
            "Error: QuantumStateCpu::add_state_with_coef(CPPCTYPE, "
            "const QuantumStateBase*): "
            "cannot add DensityMatrix to StateVector");
    }
    if (state->get_device_name() == "gpu") {
        std::cerr << "State vector on GPU cannot be added to that on CPU"
                  << std::endl;
        return;
    }
    state_add_with_coef(coef, state->data_c(), this->data_c(), this->dim);
}

void QuantumStateCpu::add_state_with_coef_single_thread(
    CPPCTYPE coef, const QuantumStateBase* state) {
    if (!state->is_state_vector()) {
        throw InoperatableQuantumStateTypeException(
            "Error: "
            "QuantumStateCpu::add_state_with_coef_single_thread(CPPCTYPE, "
            "const QuantumStateBase*): "
            "cannot add DensityMatrix to StateVector");
    }
    if (state->get_device_name() == "gpu") {
        std::cerr << "State vector on GPU cannot be added to that on CPU"
                  << std::endl;
        return;
    }
    state_add_with_coef_single_thread(
        coef, state->data_c(), this->data_c(), this->dim);
}

void QuantumStateCpu::multiply_coef(CPPCTYPE coef) {
    state_multiply(coef, this->data_c(), this->dim);
}

void QuantumStateCpu::multiply_elementwise_function(
    const std::function<CPPCTYPE(ITYPE)>& func) {
    CPPCTYPE* state = this->data_cpp();
    for (ITYPE idx = 0; idx < dim; ++idx) {
        state[idx] *= (CPPCTYPE)func(idx);
    }
}

std::vector<ITYPE> QuantumStateCpu::sampling(UINT sampling_count) {
#ifdef _USE_MPI
    if (this->outer_qc > 0) {
        // すべてのrankで同一の結果を得るために、seedを共有する
        UINT seed = random.int32();
        MPIutil& mpiutil = MPIutil::get_inst();
        if (mpiutil.get_size() > 1) mpiutil.s_u_bcast(&seed);
        return this->sampling(sampling_count, seed);
    }
#endif
    std::vector<double> stacked_prob;
    std::vector<ITYPE> result;
    double sum = 0.;
    auto ptr = this->data_cpp();
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
    return result;
}

std::vector<ITYPE> QuantumStateCpu::sampling(
    UINT sampling_count, UINT random_seed) {
    random.set_seed(random_seed);
    std::vector<ITYPE> result;

#ifdef _USE_MPI
    if (this->outer_qc > 0) {
        std::vector<double> stacked_prob;
        MPIutil& mpiutil = MPIutil::get_inst();
        UINT mpirank = mpiutil.get_rank();
        UINT mpisize = mpiutil.get_size();
        double sum = 0.;
        auto ptr = this->data_cpp();
        // resize
        stacked_prob.resize(this->dim + 1);
        result.resize(sampling_count);

        stacked_prob[0] = 0.;
        for (UINT i = 0; i < this->dim; ++i) {
            sum += norm(ptr[i]);
            stacked_prob[i + 1] = sum;
        }

        double* sumrank_prob;
        sumrank_prob = new double[mpisize];

        mpiutil.s_D_allgather(sum, sumrank_prob);

        double firstv = 0.;
        for (UINT i = 0; i < mpirank; ++i) {
            firstv += sumrank_prob[i];
        }
        for (ITYPE i = 0; i < this->dim + 1; ++i) {
            stacked_prob[i] += firstv;
        }
        delete[] sumrank_prob;

        for (UINT count = 0; count < sampling_count; ++count) {
            double r = random.uniform();
            auto ite =
                std::lower_bound(stacked_prob.begin(), stacked_prob.end(), r);
            auto index = std::distance(stacked_prob.begin(), ite) - 1;
            result[count] = index;
        }

        ITYPE geta = mpirank * this->dim;
        for (UINT i = 0; i < sampling_count; ++i) {
            if (result[i] == -1ULL or result[i] == this->dim)
                result[i] = 0ULL;
            else
                result[i] += geta;
        }
        mpiutil.m_I_allreduce(result.data(), sampling_count);
    } else
#endif
    {
        result = this->sampling(sampling_count);
    }

    return result;
}

boost::property_tree::ptree QuantumStateCpu::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "QuantumState");
    pt.put("qubit_count", _qubit_count);
    pt.put_child("classical_register", ptree::to_ptree(_classical_register));
    pt.put_child("state_vector", ptree::to_ptree(std::vector<CPPCTYPE>(
                                     _state_vector, _state_vector + _dim)));
    return pt;
}

namespace state {
CPPCTYPE inner_product(
    const QuantumState* state_bra, const QuantumState* state_ket) {
    if (state_bra->qubit_count != state_ket->qubit_count) {
        throw InvalidQubitCountException(
            "Error: inner_product(const QuantumState*, const "
            "QuantumState*): invalid qubit count");
    }
    CTYPE result;
#ifdef _USE_MPI
    if ((state_bra->outer_qc == 0) and (state_ket->outer_qc == 0))
#endif
    {
        result = state_inner_product(
            state_bra->data_c(), state_ket->data_c(), state_bra->dim);
    }
#ifdef _USE_MPI
    else {
        result = state_inner_product_mpi(state_bra->data_c(),
            state_ket->data_c(), state_bra->dim, state_ket->dim);
    }
#endif
    return result;
}
QuantumState* tensor_product(
    const QuantumState* state_left, const QuantumState* state_right) {
#ifdef _USE_MPI
    if ((state_left->outer_qc > 0) || (state_right->outer_qc > 0))
        throw NotImplementedException(
            "Error: tensor_product does not support multi-cpu");
#endif
    UINT qubit_count = state_left->qubit_count + state_right->qubit_count;
    QuantumState* qs = new QuantumState(qubit_count);
    state_tensor_product(state_left->data_c(), state_left->dim,
        state_right->data_c(), state_right->dim, qs->data_c());
    return qs;
}
QuantumState* permutate_qubit(
    const QuantumState* state, std::vector<UINT> qubit_order) {
#ifdef _USE_MPI
    if (state->outer_qc > 0)
        throw NotImplementedException(
            "Error: permutate_qubit does not support multi-cpu");
#endif
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
#ifdef _USE_MPI
    if (state->outer_qc > 0)
        throw NotImplementedException(
            "Error: drop_qubit does not support multi-cpu");
#endif
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
}  // namespace state
