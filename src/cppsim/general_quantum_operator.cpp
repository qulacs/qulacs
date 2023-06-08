#include "general_quantum_operator.hpp"

#include <Eigen/Dense>
#include <csim/stat_ops.hpp>
#include <csim/update_ops.hpp>
#include <csim/update_ops_dm.hpp>
#include <csim/utility.hpp>
#include <cstring>
#include <fstream>
#include <numeric>
#include <unsupported/Eigen/KroneckerProduct>

#include "exception.hpp"
#include "gate_factory.hpp"
#include "pauli_operator.hpp"
#include "state.hpp"
#include "type.hpp"
#include "utility.hpp"

#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif

GeneralQuantumOperator::GeneralQuantumOperator(const UINT qubit_count)
    : _qubit_count(qubit_count), _is_hermitian(true) {}

GeneralQuantumOperator::~GeneralQuantumOperator() {
    for (auto& term : this->_operator_list) {
        delete term;
    }
}

GeneralQuantumOperator::GeneralQuantumOperator(
    const GeneralQuantumOperator& obj)
    : _qubit_count(obj._qubit_count), _is_hermitian(obj._is_hermitian) {
    for (auto& pauli : obj._operator_list) {
        this->add_operator_copy(pauli);
    }
}

void GeneralQuantumOperator::add_operator(const PauliOperator* mpt) {
    GeneralQuantumOperator::add_operator_copy(mpt);
}

void GeneralQuantumOperator::add_operator_copy(const PauliOperator* mpt) {
    PauliOperator* _mpt = mpt->copy();
    GeneralQuantumOperator::add_operator_move(_mpt);
}

void GeneralQuantumOperator::add_operator_move(PauliOperator* mpt) {
    if (!check_Pauli_operator(this, mpt)) {
        throw QubitIndexOutOfRangeException(
            "Error: GeneralQuantumOperator::add_operator(const "
            "PauliOperator*): pauli_operator applies target qubit of "
            "which the index is larger than qubit_count");
    }
    if (this->_is_hermitian && std::abs(mpt->get_coef().imag()) > 0) {
        this->_is_hermitian = false;
    }
    this->_operator_list.push_back(mpt);
}

void GeneralQuantumOperator::add_operator(
    CPPCTYPE coef, std::string pauli_string) {
    PauliOperator* _mpt = new PauliOperator(pauli_string, coef);
    if (!check_Pauli_operator(this, _mpt)) {
        throw QubitIndexOutOfRangeException(
            "Error: "
            "GeneralQuantumOperator::add_operator(double,std::string):"
            " pauli_operator applies target qubit of which the index "
            "is larger than qubit_count");
    }
    if (this->_is_hermitian && std::abs(coef.imag()) > 0) {
        this->_is_hermitian = false;
    }
    this->add_operator_move(_mpt);
}

void GeneralQuantumOperator::add_operator(
    const std::vector<UINT>& target_qubit_index_list,
    const std::vector<UINT>& target_qubit_pauli_list, CPPCTYPE coef = 1.) {
    PauliOperator* _mpt = new PauliOperator(
        target_qubit_index_list, target_qubit_pauli_list, coef);
    if (!check_Pauli_operator(this, _mpt)) {
        throw QubitIndexOutOfRangeException(
            "Error: "
            "GeneralQuantumOperator::add_operator(double,std::string):"
            " pauli_operator applies target qubit of which the index "
            "is larger than qubit_count");
    }
    if (this->_is_hermitian && std::abs(coef.imag()) > 0) {
        this->_is_hermitian = false;
    }
    this->add_operator(_mpt);
    delete _mpt;
}

CPPCTYPE GeneralQuantumOperator::get_expectation_value(
    const QuantumStateBase* state) const {
    if (this->_qubit_count > state->qubit_count) {
        throw InvalidQubitCountException(
            "Error: GeneralQuantumOperator::get_expectation_value(const "
            "QuantumStateBase*): invalid qubit count");
    }

    const size_t n_terms = this->_operator_list.size();
    std::string device = state->get_device_name();
    if (device == "gpu" || device == "multi-cpu") {
        CPPCTYPE sum = 0;
        for (UINT i = 0; i < n_terms; ++i) {
            sum += _operator_list[i]->get_expectation_value(state);
        }
        return sum;
    }

    double sum_real = 0.;
    double sum_imag = 0.;
    CPPCTYPE tmp(0., 0.);
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(n_terms, 0);
#pragma omp parallel for reduction(+ : sum_real, sum_imag) private(tmp)
#endif
    for (int i = 0; i < (int)n_terms;
         ++i) {  // this variable (i) has to be signed integer because of OpenMP
                 // of Windows compiler.
        tmp = _operator_list[i]->get_expectation_value_single_thread(state);
        sum_real += tmp.real();
        sum_imag += tmp.imag();
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return CPPCTYPE(sum_real, sum_imag);
}

CPPCTYPE GeneralQuantumOperator::get_expectation_value_single_thread(
    const QuantumStateBase* state) const {
    if (this->_qubit_count > state->qubit_count) {
        std::cerr
            << "Error: GeneralQuantumOperator::get_expectation_value(const "
               "QuantumStateBase*): invalid qubit count"
            << std::endl;
        return 0.;
    }
    auto sum = std::accumulate(this->_operator_list.cbegin(),
        this->_operator_list.cend(), (CPPCTYPE)0.0,
        [&](CPPCTYPE acc, PauliOperator* pauli) {
            return acc + pauli->get_expectation_value_single_thread(state);
        });
    return sum;
}

CPPCTYPE GeneralQuantumOperator::get_transition_amplitude(
    const QuantumStateBase* state_bra,
    const QuantumStateBase* state_ket) const {
    if (this->_qubit_count > state_bra->qubit_count ||
        state_bra->qubit_count != state_ket->qubit_count) {
        throw InvalidQubitCountException(
            "Error: GeneralQuantumOperator::get_transition_amplitude(const "
            "QuantumStateBase*, const QuantumStateBase*): invalid qubit "
            "count");
    }

    auto sum = std::accumulate(this->_operator_list.cbegin(),
        this->_operator_list.cend(), (CPPCTYPE)0.0,
        [&](CPPCTYPE acc, PauliOperator* pauli) {
            return acc + pauli->get_transition_amplitude(state_bra, state_ket);
        });
    return sum;
}

void GeneralQuantumOperator::add_random_operator(const UINT operator_count) {
    const auto qubit_count = this->get_qubit_count();
    for (UINT operator_index = 0; operator_index < operator_count;
         operator_index++) {
        auto target_qubit_index_list = std::vector<UINT>(qubit_count, 0);
        auto target_qubit_pauli_list = std::vector<UINT>(qubit_count, 0);
        for (UINT qubit_index = 0; qubit_index < qubit_count; qubit_index++) {
            const UINT pauli_id = random.int32() % 4;
            target_qubit_index_list.at(qubit_index) = qubit_index;
            target_qubit_pauli_list.at(qubit_index) = pauli_id;
        }
        // -1.0 <= coef <= 1.0
        const CPPCTYPE coef = random.uniform() * 2 - 1.0;
        auto pauli_operator = PauliOperator(
            target_qubit_index_list, target_qubit_pauli_list, coef);
        this->add_operator(&pauli_operator);
    }
}

void GeneralQuantumOperator::add_random_operator(
    const UINT operator_count, UINT seed) {
    random.set_seed(seed);
    add_random_operator(operator_count);
}

CPPCTYPE
GeneralQuantumOperator::solve_ground_state_eigenvalue_by_arnoldi_method(
    QuantumStateBase* state, const UINT iter_count, const CPPCTYPE mu) const {
    if (this->get_term_count() == 0) {
        throw InvalidQuantumOperatorException(
            "Error: "
            "GeneralQuantumOperator::solve_ground_state_eigenvalue_by_"
            "arnoldi_method("
            "QuantumStateBase * state, const UINT iter_count, const "
            "CPPCTYPE mu): At least one PauliOperator is required.");
    }

    // Implemented based on
    // https://files.transtutors.com/cdn/uploadassignments/472339_1_-numerical-linear-aljebra.pdf
    const auto qubit_count = this->get_qubit_count();
    auto present_state = QuantumState(qubit_count);
    auto tmp_state = QuantumState(qubit_count);
    auto multiplied_state = QuantumState(qubit_count);
    auto mu_timed_state = QuantumState(qubit_count);

    // Vectors composing Krylov subspace.
    std::vector<QuantumStateBase*> state_list;
    state_list.reserve(iter_count + 1);
    state->normalize(state->get_squared_norm());
    state_list.push_back(state->copy());

    CPPCTYPE mu_;
    if (mu == 0.0) {
        // mu is not changed from default value.
        mu_ = this->calculate_default_mu();
    } else {
        mu_ = mu;
    }

    ComplexMatrix hessenberg_matrix =
        ComplexMatrix::Zero(iter_count, iter_count);
    for (UINT i = 0; i < iter_count; i++) {
        mu_timed_state.load(state_list[i]);
        mu_timed_state.multiply_coef(-mu_);
        this->apply_to_state(&tmp_state, *state_list[i], &multiplied_state);
        multiplied_state.add_state(&mu_timed_state);

        for (UINT j = 0; j < i + 1; j++) {
            const auto coef = state::inner_product(
                static_cast<QuantumState*>(state_list[j]), &multiplied_state);
            hessenberg_matrix(j, i) = coef;
            tmp_state.load(state_list[j]);
            tmp_state.multiply_coef(-coef);
            multiplied_state.add_state(&tmp_state);
        }

        const auto norm = multiplied_state.get_squared_norm();
        if (i != iter_count - 1) {
            hessenberg_matrix(i + 1, i) = std::sqrt(norm);
        }
        multiplied_state.normalize(norm);
        state_list.push_back(multiplied_state.copy());
    }

    Eigen::ComplexEigenSolver<ComplexMatrix> eigen_solver(hessenberg_matrix);
    const auto eigenvalues = eigen_solver.eigenvalues();
    const auto eigenvectors = eigen_solver.eigenvectors();

    // Find ground state eigenvalue and eigenvector.
    UINT minimum_eigenvalue_index = 0;
    auto minimum_eigenvalue = eigenvalues[0];
    for (UINT i = 0; i < eigenvalues.size(); i++) {
        if (eigenvalues[i].real() < minimum_eigenvalue.real()) {
            minimum_eigenvalue_index = i;
            minimum_eigenvalue = eigenvalues[i];
        }
    }

    // Compose ground state vector and store it to `state`.
    present_state.set_zero_norm_state();
    for (UINT i = 0; i < state_list.size() - 1; i++) {
        tmp_state.load(state_list[i]);
        tmp_state.multiply_coef(eigenvectors(i, minimum_eigenvalue_index));
        present_state.add_state(&tmp_state);
    }
    state->load(&present_state);

    // Free states allocated by `QuantumState::copy()`.
    for (auto used_state : state_list) {
        delete used_state;
    }
    return minimum_eigenvalue + mu_;
}

CPPCTYPE GeneralQuantumOperator::solve_ground_state_eigenvalue_by_power_method(
    QuantumStateBase* state, const UINT iter_count, const CPPCTYPE mu) const {
    if (this->get_term_count() == 0) {
        throw InvalidQuantumOperatorException(
            "Error: "
            "GeneralQuantumOperator::solve_ground_state_eigenvalue_by_"
            "power_method("
            "QuantumStateBase * state, const UINT iter_count, const "
            "CPPCTYPE mu): At least one PauliOperator is required.");
    }

    CPPCTYPE mu_;
    if (mu == 0.0) {
        // mu is not changed from default value.
        mu_ = this->calculate_default_mu();
    } else {
        mu_ = mu;
    }

    // Stores a result of A|a>
    auto multiplied_state = QuantumState(state->qubit_count);
    // Stores a result of -\mu|a>
    auto mu_timed_state = QuantumState(state->qubit_count);
    auto work_state = QuantumState(state->qubit_count);
    for (UINT i = 0; i < iter_count; i++) {
        mu_timed_state.load(state);
        mu_timed_state.multiply_coef(-mu_);

        multiplied_state.set_zero_norm_state();
        this->apply_to_state(&work_state, *state, &multiplied_state);
        state->load(&multiplied_state);
        state->add_state(&mu_timed_state);
        state->normalize(state->get_squared_norm());
    }
    return this->get_expectation_value(state) + mu;
}

void GeneralQuantumOperator::apply_to_state(QuantumStateBase* work_state,
    const QuantumStateBase& state_to_be_multiplied,
    QuantumStateBase* dst_state) const {
    if (state_to_be_multiplied.qubit_count != dst_state->qubit_count) {
        throw InvalidQubitCountException(
            "Qubit count of state_to_be_multiplied and dst_state must be the "
            "same");
    }

    dst_state->set_zero_norm_state();
    const auto term_count = this->get_term_count();
    for (UINT i = 0; i < term_count; i++) {
        work_state->load(&state_to_be_multiplied);
        const auto term = this->get_term(i);
        _apply_pauli_to_state(
            term->get_pauli_id_list(), term->get_index_list(), work_state);
        dst_state->add_state_with_coef(term->get_coef(), work_state);
    }
}

void GeneralQuantumOperator::apply_to_state(
    QuantumStateBase* state, QuantumStateBase* dst_state) const {
    if (state->qubit_count != dst_state->qubit_count) {
        throw InvalidQubitCountException(
            "Qubit count of state_to_be_multiplied and dst_state must be the "
            "same");
    }

    dst_state->set_zero_norm_state();
    const auto term_count = this->get_term_count();
    for (UINT i = 0; i < term_count; i++) {
        const auto term = this->get_term(i);
        _apply_pauli_to_state(
            term->get_pauli_id_list(), term->get_index_list(), state);
        dst_state->add_state_with_coef(term->get_coef(), state);
        _apply_pauli_to_state(
            term->get_pauli_id_list(), term->get_index_list(), state);
    }
}

void GeneralQuantumOperator::apply_to_state_single_thread(
    QuantumStateBase* state, QuantumStateBase* dst_state) const {
    if (state->qubit_count != dst_state->qubit_count) {
        throw InvalidQubitCountException(
            "Qubit count of state_to_be_multiplied and dst_state must be the "
            "same");
    }

    dst_state->set_zero_norm_state();
    const auto term_count = this->get_term_count();
    for (UINT i = 0; i < term_count; i++) {
        const auto term = this->get_term(i);
        _apply_pauli_to_state_single_thread(
            term->get_pauli_id_list(), term->get_index_list(), state);
        dst_state->add_state_with_coef_single_thread(term->get_coef(), state);
        _apply_pauli_to_state_single_thread(
            term->get_pauli_id_list(), term->get_index_list(), state);
    }
}

void GeneralQuantumOperator::_apply_pauli_to_state(
    std::vector<UINT> pauli_id_list, std::vector<UINT> target_index_list,
    QuantumStateBase* state) const {
    // this function is same as the gate::Pauli update quantum state
    if (state->is_state_vector()) {
#ifdef _USE_GPU
        if (state->get_device_name() == "gpu") {
            multi_qubit_Pauli_gate_partial_list_host(target_index_list.data(),
                pauli_id_list.data(), (UINT)target_index_list.size(),
                state->data(), state->dim, state->get_cuda_stream(),
                state->device_number);
            // _update_func_gpu(this->_target_qubit_list[0].index(), _angle,
            // state->data(), state->dim);
            return;
        }
#endif
        multi_qubit_Pauli_gate_partial_list(target_index_list.data(),
            pauli_id_list.data(), (UINT)target_index_list.size(),
            state->data_c(), state->dim);
    } else {
        dm_multi_qubit_Pauli_gate_partial_list(target_index_list.data(),
            pauli_id_list.data(), (UINT)target_index_list.size(),
            state->data_c(), state->dim);
    }
}

void GeneralQuantumOperator::_apply_pauli_to_state_single_thread(
    std::vector<UINT> pauli_id_list, std::vector<UINT> target_index_list,
    QuantumStateBase* state) const {
    // this function is same as the gate::Pauli update quantum state
    if (state->is_state_vector()) {
#ifdef _USE_GPU
        if (state->get_device_name() == "gpu") {
            // TODO: make it single thread for this function
            multi_qubit_Pauli_gate_partial_list_host(target_index_list.data(),
                pauli_id_list.data(), (UINT)target_index_list.size(),
                state->data(), state->dim, state->get_cuda_stream(),
                state->device_number);
            // _update_func_gpu(this->_target_qubit_list[0].index(), _angle,
            // state->data(), state->dim);
            return;
        }
#endif
        multi_qubit_Pauli_gate_partial_list_single_thread(
            target_index_list.data(), pauli_id_list.data(),
            (UINT)target_index_list.size(), state->data_c(), state->dim);
    } else {
        throw std::runtime_error(
            "apply single thread is not implemented for density matrix");
    }
}

CPPCTYPE GeneralQuantumOperator::calculate_default_mu() const {
    double mu = 0.0;
    const auto term_count = this->get_term_count();
    for (UINT i = 0; i < term_count; i++) {
        const auto term = this->get_term(i);
        mu += std::abs(term->get_coef().real());
    }
    return static_cast<CPPCTYPE>(mu);
}

GeneralQuantumOperator* GeneralQuantumOperator::copy() const {
    auto quantum_operator = new GeneralQuantumOperator(_qubit_count);
    for (auto pauli : this->_operator_list) {
        quantum_operator->add_operator_copy(pauli);
    }
    return quantum_operator;
}

SparseComplexMatrixRowMajor _tensor_product(
    const std::vector<SparseComplexMatrixRowMajor>& _obs) {
    std::vector<SparseComplexMatrixRowMajor> obs(_obs);
    int sz = obs.size();
    while (sz != 1) {
        if (sz % 2 == 0) {
            for (int i = 0; i < sz; i += 2) {
                obs[i >> 1] =
                    Eigen::kroneckerProduct(obs[i], obs[i + 1]).eval();
            }
            sz >>= 1;
        } else {
            obs[sz - 2] =
                Eigen::kroneckerProduct(obs[sz - 2], obs[sz - 1]).eval();
            sz--;
        }
    }
    return obs[0];
}

SparseComplexMatrixRowMajor GeneralQuantumOperator::get_matrix() const {
    SparseComplexMatrixRowMajor sigma_x(2, 2), sigma_y(2, 2), sigma_z(2, 2),
        sigma_i(2, 2);
    sigma_x.insert(0, 1) = 1.0, sigma_x.insert(1, 0) = 1.0;
    sigma_y.insert(0, 1) = -1.0i, sigma_y.insert(1, 0) = 1.0i;
    sigma_z.insert(0, 0) = 1.0, sigma_z.insert(1, 1) = -1.0;
    sigma_i.insert(0, 0) = 1.0, sigma_i.insert(1, 1) = 1.0;
    std::vector<SparseComplexMatrixRowMajor> pauli_matrix_list = {
        sigma_i, sigma_x, sigma_y, sigma_z};

    int n_terms = this->get_term_count();
    int n_qubits = this->get_qubit_count();
    SparseComplexMatrixRowMajor hamiltonian_matrix(
        1 << n_qubits, 1 << n_qubits);
    hamiltonian_matrix.setZero();
#ifdef _OPENMP
#pragma omp declare reduction(+ : SparseComplexMatrixRowMajor : omp_out += \
                                      omp_in) initializer(omp_priv = omp_orig)
#pragma omp parallel for reduction(+ : hamiltonian_matrix)
#endif
    for (int i = 0; i < n_terms; i++) {
        auto const pauli = this->get_term(i);
        auto const pauli_id_list = pauli->get_pauli_id_list();
        auto const pauli_target_list = pauli->get_index_list();

        std::vector<SparseComplexMatrixRowMajor>
            init_hamiltonian_pauli_matrix_list(
                n_qubits, sigma_i);  // initialize matrix_list I
        for (int j = 0; j < (int)pauli_target_list.size(); j++) {
            init_hamiltonian_pauli_matrix_list[pauli_target_list[j]] =
                pauli_matrix_list[pauli_id_list[j]];  // ex) [X,X,I,I]
        }
        std::reverse(init_hamiltonian_pauli_matrix_list.begin(),
            init_hamiltonian_pauli_matrix_list.end());
        hamiltonian_matrix +=
            pauli->get_coef() *
            _tensor_product(init_hamiltonian_pauli_matrix_list);
    }
    hamiltonian_matrix.prune(CPPCTYPE(0, 0));
    return hamiltonian_matrix;
}

GeneralQuantumOperator* GeneralQuantumOperator::get_dagger() const {
    auto quantum_operator = new GeneralQuantumOperator(_qubit_count);
    for (auto pauli : this->_operator_list) {
        quantum_operator->add_operator(
            std::conj(pauli->get_coef()), pauli->get_pauli_string());
    }
    return quantum_operator;
}

boost::property_tree::ptree GeneralQuantumOperator::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "GeneralQuantumOperator");
    pt.put("qubit_count", _qubit_count);
    std::vector<boost::property_tree::ptree> operator_list_pt;
    std::transform(_operator_list.begin(), _operator_list.end(),
        std::back_inserter(operator_list_pt),
        [](const PauliOperator* po) { return po->to_ptree(); });
    pt.put_child("operator_list", ptree::to_ptree(operator_list_pt));
    return pt;
}

GeneralQuantumOperator GeneralQuantumOperator::operator+(
    const GeneralQuantumOperator& target) const {
    return GeneralQuantumOperator(*this) += target;
}

GeneralQuantumOperator GeneralQuantumOperator::operator+(
    const PauliOperator& target) const {
    return GeneralQuantumOperator(*this) += target;
}

GeneralQuantumOperator& GeneralQuantumOperator::operator+=(
    const GeneralQuantumOperator& target) {
    ITYPE i, j;
    auto terms = target.get_terms();
    // #pragma omp parallel for
    for (i = 0; i < _operator_list.size(); i++) {
        auto pauli_operator = _operator_list[i];
        for (j = 0; j < terms.size(); j++) {
            auto target_operator = terms[j];
            auto pauli_x = pauli_operator->get_x_bits();
            auto pauli_z = pauli_operator->get_z_bits();
            auto target_x = target_operator->get_x_bits();
            auto target_z = target_operator->get_z_bits();
            if (pauli_x.size() != target_x.size()) {
                UINT max_size = std::max(pauli_x.size(), target_x.size());
                pauli_x.resize(max_size);
                pauli_z.resize(max_size);
                target_x.resize(max_size);
                target_z.resize(max_size);
            }
            if (pauli_x == target_x && pauli_z == target_z) {
                _operator_list[i]->change_coef(_operator_list[i]->get_coef() +
                                               target_operator->get_coef());
            }
        }
    }
    for (j = 0; j < terms.size(); j++) {
        auto target_operator = terms[j];
        bool flag = true;
        for (i = 0; i < _operator_list.size(); i++) {
            auto pauli_operator = _operator_list[i];
            auto pauli_x = pauli_operator->get_x_bits();
            auto pauli_z = pauli_operator->get_z_bits();
            auto target_x = target_operator->get_x_bits();
            auto target_z = target_operator->get_z_bits();
            if (pauli_x.size() != target_x.size()) {
                UINT max_size = std::max(pauli_x.size(), target_x.size());
                pauli_x.resize(max_size);
                pauli_z.resize(max_size);
                target_x.resize(max_size);
                target_z.resize(max_size);
            }
            if (pauli_x == target_x && pauli_z == target_z) {
                flag = false;
            }
        }
        if (flag) {
            this->add_operator_copy(target_operator);
        }
    }
    return *this;
}

GeneralQuantumOperator& GeneralQuantumOperator::operator+=(
    const PauliOperator& target) {
    bool flag = true;
    ITYPE i;
    // #pragma omp parallel for
    for (i = 0; i < _operator_list.size(); i++) {
        auto pauli_operator = _operator_list[i];
        auto pauli_x = pauli_operator->get_x_bits();
        auto pauli_z = pauli_operator->get_z_bits();
        auto target_x = target.get_x_bits();
        auto target_z = target.get_z_bits();
        if (pauli_x.size() != target_x.size()) {
            UINT max_size = std::max(pauli_x.size(), target_x.size());
            pauli_x.resize(max_size);
            pauli_z.resize(max_size);
            target_x.resize(max_size);
            target_z.resize(max_size);
        }
        if (pauli_x == target_x && pauli_z == target_z) {
            _operator_list[i]->change_coef(
                _operator_list[i]->get_coef() + target.get_coef());
            flag = false;
        }
    }
    if (flag) {
        this->add_operator_copy(&target);
    }
    return *this;
}

GeneralQuantumOperator GeneralQuantumOperator::operator-(
    const GeneralQuantumOperator& target) const {
    return GeneralQuantumOperator(*this) -= target;
}

GeneralQuantumOperator GeneralQuantumOperator::operator-(
    const PauliOperator& target) const {
    return GeneralQuantumOperator(*this) -= target;
}

GeneralQuantumOperator& GeneralQuantumOperator::operator-=(
    const GeneralQuantumOperator& target) {
    ITYPE i, j;
    auto terms = target.get_terms();
    // #pragma omp parallel for
    for (i = 0; i < _operator_list.size(); i++) {
        auto pauli_operator = _operator_list[i];
        for (j = 0; j < terms.size(); j++) {
            auto target_operator = terms[j];
            auto pauli_x = pauli_operator->get_x_bits();
            auto pauli_z = pauli_operator->get_z_bits();
            auto target_x = target_operator->get_x_bits();
            auto target_z = target_operator->get_z_bits();
            if (pauli_x.size() != target_x.size()) {
                UINT max_size = std::max(pauli_x.size(), target_x.size());
                pauli_x.resize(max_size);
                pauli_z.resize(max_size);
                target_x.resize(max_size);
                target_z.resize(max_size);
            }
            if (pauli_x == target_x && pauli_z == target_z) {
                _operator_list[i]->change_coef(_operator_list[i]->get_coef() -
                                               target_operator->get_coef());
            }
        }
    }
    for (j = 0; j < terms.size(); j++) {
        auto target_operator = terms[j];
        bool flag = true;
        for (i = 0; i < _operator_list.size(); i++) {
            auto pauli_operator = _operator_list[i];
            auto pauli_x = pauli_operator->get_x_bits();
            auto pauli_z = pauli_operator->get_z_bits();
            auto target_x = target_operator->get_x_bits();
            auto target_z = target_operator->get_z_bits();
            if (pauli_x.size() != target_x.size()) {
                UINT max_size = std::max(pauli_x.size(), target_x.size());
                pauli_x.resize(max_size);
                pauli_z.resize(max_size);
                target_x.resize(max_size);
                target_z.resize(max_size);
            }
            if (pauli_x == target_x && pauli_z == target_z) {
                flag = false;
            }
        }
        if (flag) {
            auto copy = target_operator->copy();
            copy->change_coef(-copy->get_coef());
            this->add_operator_move(copy);
        }
    }
    return *this;
}

GeneralQuantumOperator& GeneralQuantumOperator::operator-=(
    const PauliOperator& target) {
    bool flag = true;
    ITYPE i;
    for (i = 0; i < _operator_list.size(); i++) {
        auto pauli_operator = _operator_list[i];
        auto pauli_x = pauli_operator->get_x_bits();
        auto pauli_z = pauli_operator->get_z_bits();
        auto target_x = target.get_x_bits();
        auto target_z = target.get_z_bits();
        if (pauli_x.size() != target_x.size()) {
            UINT max_size = std::max(pauli_x.size(), target_x.size());
            pauli_x.resize(max_size);
            pauli_z.resize(max_size);
            target_x.resize(max_size);
            target_z.resize(max_size);
        }
        if (pauli_x == target_x && pauli_z == target_z) {
            _operator_list[i]->change_coef(
                _operator_list[i]->get_coef() - target.get_coef());
            flag = false;
        }
    }
    if (flag) {
        auto copy = target.copy();
        copy->change_coef(-copy->get_coef());
        this->add_operator_move(copy);
    }
    return *this;
}
GeneralQuantumOperator GeneralQuantumOperator::operator*(
    const GeneralQuantumOperator& target) const {
    return GeneralQuantumOperator(*this) *= target;
}

GeneralQuantumOperator GeneralQuantumOperator::operator*(
    const PauliOperator& target) const {
    return GeneralQuantumOperator(*this) *= target;
}

GeneralQuantumOperator GeneralQuantumOperator::operator*(
    CPPCTYPE target) const {
    return GeneralQuantumOperator(*this) *= target;
}

GeneralQuantumOperator& GeneralQuantumOperator::operator*=(
    const GeneralQuantumOperator& target) {
    auto this_copy = this->copy();
    auto terms = this_copy->get_terms();
    auto target_terms = target.get_terms();

    // initialize (*this) operator to empty.
    for (auto& term : _operator_list) {
        delete term;
    }
    _operator_list.clear();

    ITYPE i, j;
    // #pragma omp parallel for
    for (i = 0; i < terms.size(); i++) {
        auto pauli_operator = terms[i];
        for (j = 0; j < target_terms.size(); j++) {
            auto target_operator = target_terms[j];
            PauliOperator product = (*pauli_operator) * (*target_operator);
            *this += product;
        }
    }
    delete this_copy;
    return *this;
}

GeneralQuantumOperator& GeneralQuantumOperator::operator*=(
    const PauliOperator& target) {
    auto this_copy = this->copy();
    ITYPE i;
    auto terms = this_copy->get_terms();

    // initialize (*this) operator to empty.
    for (auto& term : _operator_list) {
        delete term;
    }
    _operator_list.clear();

    // #pragma omp parallel for
    for (i = 0; i < terms.size(); i++) {
        auto pauli_operator = terms[i];
        PauliOperator product = (*pauli_operator) * (target);
        *this += product;
    }
    delete this_copy;
    return *this;
}

GeneralQuantumOperator& GeneralQuantumOperator::operator*=(CPPCTYPE target) {
    ITYPE i;
    // #pragma omp parallel for
    for (i = 0; i < _operator_list.size(); i++) {
        *_operator_list[i] *= target;
    }
    return *this;
}
namespace quantum_operator {
GeneralQuantumOperator* create_general_quantum_operator_from_openfermion_file(
    std::string file_path) {
    UINT qubit_count = 0;
    std::vector<CPPCTYPE> coefs;
    std::vector<std::string> ops;

    // loading lines and check qubit_count
    std::string str_buf;
    std::vector<std::string> index_list;

    std::ifstream ifs;
    std::string line;
    ifs.open(file_path);

    while (getline(ifs, line)) {
        std::tuple<double, double, std::string> parsed_items =
            parse_openfermion_line(line);
        const auto coef_real = std::get<0>(parsed_items);
        const auto coef_imag = std::get<1>(parsed_items);
        str_buf = std::get<2>(parsed_items);

        CPPCTYPE coef(coef_real, coef_imag);
        coefs.push_back(coef);
        ops.push_back(str_buf);
        index_list = split(str_buf, "IXYZ ");

        for (UINT i = 0; i < index_list.size(); ++i) {
            UINT n = std::stoi(index_list[i]) + 1;
            if (qubit_count < n) qubit_count = n;
        }
    }
    if (!ifs.eof()) {
        throw InvalidOpenfermionFormatException("ERROR: Invalid format");
    }
    ifs.close();

    GeneralQuantumOperator* general_quantum_operator =
        new GeneralQuantumOperator(qubit_count);

    for (UINT i = 0; i < ops.size(); ++i) {
        general_quantum_operator->add_operator(coefs[i], ops[i].c_str());
    }

    return general_quantum_operator;
}

GeneralQuantumOperator* create_general_quantum_operator_from_openfermion_text(
    std::string text) {
    UINT qubit_count = 0;
    std::vector<CPPCTYPE> coefs;
    std::vector<std::string> ops;

    std::string str_buf;
    std::vector<std::string> index_list;

    std::vector<std::string> lines;
    lines = split(text, "\n");
    for (std::string line : lines) {
        std::tuple<double, double, std::string> parsed_items =
            parse_openfermion_line(line);
        const auto coef_real = std::get<0>(parsed_items);
        const auto coef_imag = std::get<1>(parsed_items);
        str_buf = std::get<2>(parsed_items);

        CPPCTYPE coef(coef_real, coef_imag);
        coefs.push_back(coef);
        ops.push_back(str_buf);
        index_list = split(str_buf, "IXYZ ");

        for (UINT i = 0; i < index_list.size(); ++i) {
            UINT n = std::stoi(index_list[i]) + 1;
            if (qubit_count < n) qubit_count = n;
        }
    }
    GeneralQuantumOperator* general_quantum_operator =
        new GeneralQuantumOperator(qubit_count);

    for (UINT i = 0; i < ops.size(); ++i) {
        general_quantum_operator->add_operator(coefs[i], ops[i].c_str());
    }

    return general_quantum_operator;
}

std::pair<GeneralQuantumOperator*, GeneralQuantumOperator*>
create_split_general_quantum_operator(std::string file_path) {
    UINT qubit_count = 0;
    std::vector<CPPCTYPE> coefs;
    std::vector<std::string> ops;

    std::ifstream ifs;
    ifs.open(file_path);

    if (!ifs) {
        throw IOException("ERROR: Cannot open file");
    }

    // loading lines and check qubit_count
    std::string str_buf;
    std::vector<std::string> index_list;

    std::string line;
    while (getline(ifs, line)) {
        std::tuple<double, double, std::string> parsed_items =
            parse_openfermion_line(line);
        const auto coef_real = std::get<0>(parsed_items);
        const auto coef_imag = std::get<1>(parsed_items);
        str_buf = std::get<2>(parsed_items);
        if (str_buf == (std::string)NULL) {
            continue;
        }
        CPPCTYPE coef(coef_real, coef_imag);
        coefs.push_back(coef);
        ops.push_back(str_buf);
        index_list = split(str_buf, "IXYZ ");

        for (UINT i = 0; i < index_list.size(); ++i) {
            UINT n = std::stoi(index_list[i]) + 1;
            if (qubit_count < n) qubit_count = n;
        }
    }
    if (!ifs.eof()) {
        throw InvalidOpenfermionFormatException("ERROR: Invalid format");
    }
    ifs.close();

    GeneralQuantumOperator* general_quantum_operator_diag =
        new GeneralQuantumOperator(qubit_count);
    GeneralQuantumOperator* general_quantum_operator_non_diag =
        new GeneralQuantumOperator(qubit_count);

    for (UINT i = 0; i < ops.size(); ++i) {
        if (ops[i].find("X") != std::string::npos ||
            ops[i].find("Y") != std::string::npos) {
            general_quantum_operator_non_diag->add_operator(
                coefs[i], ops[i].c_str());
        } else {
            general_quantum_operator_diag->add_operator(
                coefs[i], ops[i].c_str());
        }
    }

    return std::make_pair(
        general_quantum_operator_diag, general_quantum_operator_non_diag);
}

SinglePauliOperator* single_pauli_operator_from_ptree(
    const boost::property_tree::ptree& pt) {
    std::string name = pt.get<std::string>("name");
    if (name != "SinglePauliOperator") {
        throw UnknownPTreePropertyValueException(
            "unknown value for property \"name\":" + name);
    }
    UINT index = pt.get<UINT>("index");
    UINT pauli_id = pt.get<UINT>("pauli_id");
    return new SinglePauliOperator(index, pauli_id);
}

PauliOperator* pauli_operator_from_ptree(
    const boost::property_tree::ptree& pt) {
    std::string name = pt.get<std::string>("name");
    if (name != "PauliOperator") {
        throw UnknownPTreePropertyValueException(
            "unknown value for property \"name\":" + name);
    }
    std::vector<boost::property_tree::ptree> pauli_list_pt =
        ptree::ptree_array_from_ptree(pt.get_child("pauli_list"));
    CPPCTYPE coef = ptree::complex_from_ptree(pt.get_child("coef"));
    PauliOperator* po = new PauliOperator(coef);
    for (const boost::property_tree::ptree& pauli_pt : pauli_list_pt) {
        SinglePauliOperator* spo = single_pauli_operator_from_ptree(pauli_pt);
        po->add_single_Pauli(spo->index(), spo->pauli_id());
        free(spo);
    }
    return po;
}

GeneralQuantumOperator* from_ptree(const boost::property_tree::ptree& pt) {
    std::string name = pt.get<std::string>("name");
    if (name != "GeneralQuantumOperator") {
        throw UnknownPTreePropertyValueException(
            "unknown value for property \"name\":" + name);
    }
    UINT qubit_count = pt.get<UINT>("qubit_count");
    std::vector<boost::property_tree::ptree> operator_list_pt =
        ptree::ptree_array_from_ptree(pt.get_child("operator_list"));
    GeneralQuantumOperator* gqo = new GeneralQuantumOperator(qubit_count);
    for (const boost::property_tree::ptree& operator_pt : operator_list_pt) {
        gqo->add_operator_move(pauli_operator_from_ptree(operator_pt));
    }
    return gqo;
}
}  // namespace quantum_operator

bool check_Pauli_operator(const GeneralQuantumOperator* quantum_operator,
    const PauliOperator* pauli_operator) {
    auto vec = pauli_operator->get_index_list();
    UINT val = 0;
    if (vec.size() > 0) {
        val = std::max(val, *std::max_element(vec.begin(), vec.end()));
    }
    return val < (quantum_operator->get_qubit_count());
}

std::string GeneralQuantumOperator::to_string() const {
    std::stringstream os;
    auto term_count = this->get_term_count();
    for (UINT index = 0; index < term_count; index++) {
        os << this->get_term(index)->get_coef() << " ";
        os << this->get_term(index)->get_pauli_string();
        if (index != term_count - 1) {
            os << " + ";
        }
    }
    return os.str();
}
