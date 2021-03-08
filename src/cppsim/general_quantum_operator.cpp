#include <cstring>
#include <fstream>
#include <numeric>

#include "type.hpp"
#include "utility.hpp"

#ifndef _MSC_VER
extern "C" {
#include <csim/stat_ops.h>
}
#else
#include <csim/stat_ops.h>
#endif
#include <Eigen/Dense>

#include "gate_factory.hpp"
#include "general_quantum_operator.hpp"
#include "pauli_operator.hpp"
#include "state.hpp"

GeneralQuantumOperator::GeneralQuantumOperator(const UINT qubit_count)
    : _qubit_count(qubit_count), _is_hermitian(true) {}

GeneralQuantumOperator::~GeneralQuantumOperator() {
    for (auto& term : this->_operator_list) {
        delete term;
    }
}

void GeneralQuantumOperator::add_operator(const PauliOperator* mpt) {
    PauliOperator* _mpt = mpt->copy();
    if (!check_Pauli_operator(this, _mpt)) {
        std::cerr << "Error: GeneralQuantumOperator::add_operator(const "
                     "PauliOperator*): pauli_operator applies target qubit of "
                     "which the index is larger than qubit_count"
                  << std::endl;
        return;
    }
    if (this->_is_hermitian && std::abs(_mpt->get_coef().imag()) > 0) {
        this->_is_hermitian = false;
    }
    this->_operator_list.push_back(_mpt);
}

void GeneralQuantumOperator::add_operator(
    CPPCTYPE coef, std::string pauli_string) {
    PauliOperator* _mpt = new PauliOperator(pauli_string, coef);
    if (!check_Pauli_operator(this, _mpt)) {
        std::cerr << "Error: "
                     "GeneralQuantumOperator::add_operator(double,std::string):"
                     " pauli_operator applies target qubit of which the index "
                     "is larger than qubit_count"
                  << std::endl;
        return;
    }
    if (this->_is_hermitian && std::abs(coef.imag()) > 0) {
        this->_is_hermitian = false;
    }
    this->add_operator(_mpt);
    delete _mpt;
}

CPPCTYPE GeneralQuantumOperator::get_expectation_value(
    const QuantumStateBase* state) const {
    if (this->_qubit_count != state->qubit_count) {
        std::cerr
            << "Error: GeneralQuantumOperator::get_expectation_value(const "
               "QuantumStateBase*): invalid qubit count"
            << std::endl;
        return 0.;
    }
    auto sum = std::accumulate(this->_operator_list.cbegin(),
        this->_operator_list.cend(), (CPPCTYPE)0.0,
        [&](CPPCTYPE acc, PauliOperator* pauli) {
            return acc + pauli->get_expectation_value(state);
        });
    return sum;
}

CPPCTYPE GeneralQuantumOperator::get_transition_amplitude(
    const QuantumStateBase* state_bra,
    const QuantumStateBase* state_ket) const {
    if (this->_qubit_count != state_bra->qubit_count ||
        this->_qubit_count != state_ket->qubit_count) {
        std::cerr
            << "Error: GeneralQuantumOperator::get_transition_amplitude(const "
               "QuantumStateBase*, const QuantumStateBase*): invalid qubit "
               "count"
            << std::endl;
        return 0.;
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
    Random random;
    for (UINT operator_index = 0; operator_index < operator_count;
        operator_index++) {
        auto target_qubit_index_list = std::vector<UINT>(qubit_count, 0);
        auto target_qubit_pauli_list = std::vector<UINT>(qubit_count, 0);
        for (UINT qubit_index = 0; qubit_index < qubit_count; qubit_index++) {
            const UINT pauli_id = random.int32() % 4;
            target_qubit_index_list.at(qubit_index) = qubit_index;
            target_qubit_pauli_list.at(qubit_index) = pauli_id;
        }
        const CPPCTYPE coef = random.uniform();
        auto pauli_operator = PauliOperator(
            target_qubit_index_list, target_qubit_pauli_list, coef);
        this->add_operator(&pauli_operator);
    }
}


CPPCTYPE
GeneralQuantumOperator::solve_ground_state_eigenvalue_by_arnoldi_method(
    QuantumStateBase* state, const UINT iter_count, const CPPCTYPE mu) const {
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
    const auto minimum_eigenvalue = this->calculate_ground_state_eigenvector(
        eigenvalues, eigenvectors, state_list, state, &tmp_state);

    // Free states allocated by `QuantumState::copy()`.
    for (auto used_state : state_list) {
        delete used_state;
    }
    return minimum_eigenvalue + mu_;
}

CPPCTYPE GeneralQuantumOperator::solve_ground_state_eigenvalue_by_power_method(
    QuantumStateBase* state, const UINT iter_count, const CPPCTYPE mu) const {
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

        multiplied_state.multiply_coef(0.0);
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
        throw std::invalid_argument(
            "Qubit count of state_to_be_multiplied and dst_state must be the "
            "same");
    }

    dst_state->multiply_coef(0.0);
    const auto term_count = this->get_term_count();
    for (UINT i = 0; i < term_count; i++) {
        work_state->load(&state_to_be_multiplied);
        const auto term = this->get_term(i);
        auto pauli_operator =
            gate::Pauli(term->get_index_list(), term->get_pauli_id_list());
        pauli_operator->update_quantum_state(work_state);
        work_state->multiply_coef(term->get_coef());
        dst_state->add_state(work_state);
        delete pauli_operator;
    }
}

CPPCTYPE GeneralQuantumOperator::calculate_default_mu() const {
    CPPCTYPE mu = 0.0;
    const auto term_count = this->get_term_count();
    for (UINT i = 0; i < term_count; i++) {
        const auto term = this->get_term(i);
        mu += term->get_coef();
    }
    return mu;
}

CPPCTYPE GeneralQuantumOperator::calculate_ground_state_eigenvector(
    const Eigen::VectorXcd& eigenvalues, const ComplexMatrix& eigenvectors,
    const std::vector<QuantumStateBase*>& state_list, QuantumStateBase* state,
    QuantumStateBase* tmp_state) const {
    // Find ground state eigenvalue.
    UINT minimum_eigenvalue_index = 0;
    auto minimum_eigenvalue = eigenvalues[0];
    for (UINT i = 0; i < eigenvalues.size(); i++) {
        if (eigenvalues[i].real() < minimum_eigenvalue.real()) {
            minimum_eigenvalue_index = i;
            minimum_eigenvalue = eigenvalues[i];
        }
    }

    // Compose ground state eigenvector and store it to `state`.
    state->multiply_coef(0.0);
    for (UINT i = 0; i < state_list.size() - 1; i++) {
        tmp_state->load(state_list[i]);
        tmp_state->multiply_coef(eigenvectors(i, minimum_eigenvalue_index));
        state->add_state(tmp_state);
    }
    return minimum_eigenvalue;
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
        std::cerr << "ERROR: Invalid format" << std::endl;
        return (GeneralQuantumOperator*)NULL;
    }
    ifs.close();

    GeneralQuantumOperator* general_quantum_operator =
        new GeneralQuantumOperator(qubit_count);

    for (UINT i = 0; i < ops.size(); ++i) {
        general_quantum_operator->add_operator(
            new PauliOperator(ops[i].c_str(), coefs[i]));
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
        general_quantum_operator->add_operator(
            new PauliOperator(ops[i].c_str(), coefs[i]));
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
        std::cerr << "ERROR: Cannot open file" << std::endl;
        return std::make_pair(
            (GeneralQuantumOperator*)NULL, (GeneralQuantumOperator*)NULL);
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
        std::cerr << "ERROR: Invalid format" << std::endl;
        return std::make_pair(
            (GeneralQuantumOperator*)NULL, (GeneralQuantumOperator*)NULL);
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
                new PauliOperator(ops[i].c_str(), coefs[i]));
        } else {
            general_quantum_operator_diag->add_operator(
                new PauliOperator(ops[i].c_str(), coefs[i]));
        }
    }

    return std::make_pair(
        general_quantum_operator_diag, general_quantum_operator_non_diag);
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