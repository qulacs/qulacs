#include "observable.hpp"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#undef NDEBUG
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

#include "state.hpp"
#include "utility.hpp"

void HermitianQuantumOperator::add_operator(const PauliOperator* mpt) {
    if (std::abs(mpt->get_coef().imag()) > 0) {
        std::cerr << "Error: HermitianQuantumOperator::add_operator(const "
                     "PauliOperator* mpt): PauliOperator must be Hermitian."
                  << std::endl;
        return;
    }
    GeneralQuantumOperator::add_operator(mpt);
}

void HermitianQuantumOperator::add_operator(
    CPPCTYPE coef, std::string pauli_string) {
    if (std::abs(coef.imag()) > 0) {
        std::cerr << "Error: HermitianQuantumOperator::add_operator(const "
                     "PauliOperator* mpt): PauliOperator must be Hermitian."
                  << std::endl;
        return;
    }
    GeneralQuantumOperator::add_operator(coef, pauli_string);
}

CPPCTYPE HermitianQuantumOperator::get_expectation_value(
    const QuantumStateBase* state) const {
    return GeneralQuantumOperator::get_expectation_value(state).real();
}

CPPCTYPE
HermitianQuantumOperator::solve_ground_state_eigenvalue_by_lanczos_method(
    QuantumStateBase* state, const UINT iter_count, const CPPCTYPE mu) const {
    if (this->get_term_count() == 0) {
        std::cerr << "Error: "
                     "HermitianQuantumOperator::solve_ground_state_eigenvalue_"
                     "by_lanczos_method("
                     "QuantumStateBase * state, const UINT iter_count, const "
                     "CPPCTYPE mu): At least one PauliOperator is required.";
        return 0;
    }

    const auto qubit_count = this->get_qubit_count();
    QuantumState tmp_state(qubit_count);
    QuantumState multiplied_state(qubit_count);
    QuantumState mu_timed_state(qubit_count);

    std::vector<QuantumStateBase*> state_list;
    state->normalize(state->get_squared_norm());
    state_list.push_back(state->copy());

    CPPCTYPE mu_;
    if (mu == 0.0) {
        // mu is not changed from default value.
        mu_ = this->calculate_default_mu();
    } else {
        mu_ = mu;
    }

    Eigen::VectorXd alpha_v(iter_count);
    Eigen::VectorXd beta_v(iter_count - 1);
    for (UINT i = 0; i < iter_count; i++) {
        // v = (A - μI) * q_i
        mu_timed_state.load(state_list[i]);
        mu_timed_state.multiply_coef(-mu_);
        this->apply_to_state(&tmp_state, *state_list[i], &multiplied_state);
        multiplied_state.add_state(&mu_timed_state);
        // α = q_i^T * v
        const auto alpha = state::inner_product(
            static_cast<QuantumState*>(state_list[i]), &multiplied_state);
        alpha_v(i) = alpha.real();
        // In the last iteration, no need to calculate β.
        if (i == iter_count - 1) {
            break;
        }

        tmp_state.load(state_list[i]);
        tmp_state.multiply_coef(-alpha);
        // v -= α_i * q_i
        multiplied_state.add_state(&tmp_state);
        if (i != 0) {
            tmp_state.load(state_list[i - 1]);
            tmp_state.multiply_coef(-beta_v(i - 1));
            // v -= β_{i-1} * q_{i-1}
            multiplied_state.add_state(&tmp_state);
        }

        const auto beta = std::sqrt(multiplied_state.get_squared_norm());
        beta_v(i) = beta;
        multiplied_state.multiply_coef(1 / beta);
        state_list.push_back(multiplied_state.copy());
    }

    Eigen::SelfAdjointEigenSolver<ComplexMatrix> solver;
    solver.computeFromTridiagonal(alpha_v, beta_v);
    const auto eigenvalues = solver.eigenvalues();
    // Find ground state eigenvalue and eigenvector.
    UINT minimum_eigenvalue_index = 0;
    auto minimum_eigenvalue = eigenvalues[0];
    for (UINT i = 0; i < eigenvalues.size(); i++) {
        if (eigenvalues[i] < minimum_eigenvalue) {
            minimum_eigenvalue_index = i;
            minimum_eigenvalue = eigenvalues(i);
        }
    }

    // Compose ground state vector.
    auto eigenvectors = solver.eigenvectors();
    auto eigenvector_in_krylov = eigenvectors.col(minimum_eigenvalue_index);
    // Store ground state eigenvector to `state`.
    state->multiply_coef(0.0);
    for (UINT i = 0; i < state_list.size(); i++) {
        tmp_state.load(state_list[i]);
        tmp_state.multiply_coef(eigenvector_in_krylov(i));
        state->add_state(&tmp_state);
    }

    // Free states allocated by `QuantumState::copy()`.
    for (auto used_state : state_list) {
        delete used_state;
    }

    return minimum_eigenvalue + mu_;
}

std::string HermitianQuantumOperator::to_string() const {
    std::stringstream os;
    auto term_count = this->get_term_count();
    for (UINT index = 0; index < term_count; index++) {
        os << this->get_term(index)->get_coef().real() << " ";
        os << this->get_term(index)->get_pauli_string();
        if (index != term_count - 1) {
            os << " + ";
        }
    }
    return os.str();
}

namespace observable {
HermitianQuantumOperator* create_observable_from_openfermion_file(
    std::string file_path) {
    UINT qubit_count = 0;
    std::vector<CPPCTYPE> coefs;
    std::vector<std::string> ops;

    std::ifstream ifs;
    ifs.open(file_path);

    if (!ifs) {
        std::cerr << "ERROR: Cannot open file" << std::endl;
        return NULL;
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
        return NULL;
    }
    ifs.close();

    HermitianQuantumOperator* observable =
        new HermitianQuantumOperator(qubit_count);

    for (UINT i = 0; i < ops.size(); ++i) {
        observable->add_operator(new PauliOperator(ops[i].c_str(), coefs[i]));
    }

    return observable;
}

HermitianQuantumOperator* create_observable_from_openfermion_text(
    const std::string& text) {
    UINT qubit_count = 0;
    std::vector<CPPCTYPE> coefs;
    std::vector<std::string> ops;

    std::vector<std::string> lines;
    std::string str_buf;
    std::vector<std::string> index_list;

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
    HermitianQuantumOperator* hermitian_quantum_operator =
        new HermitianQuantumOperator(qubit_count);

    for (UINT i = 0; i < ops.size(); ++i) {
        hermitian_quantum_operator->add_operator(
            new PauliOperator(ops[i].c_str(), coefs[i]));
    }

    return hermitian_quantum_operator;
}

std::pair<HermitianQuantumOperator*, HermitianQuantumOperator*>
create_split_observable(std::string file_path) {
    UINT qubit_count = 0;
    std::vector<CPPCTYPE> coefs;
    std::vector<std::string> ops;

    std::ifstream ifs;
    ifs.open(file_path);

    if (!ifs) {
        std::cerr << "ERROR: Cannot open file" << std::endl;
        return std::make_pair(
            (HermitianQuantumOperator*)NULL, (HermitianQuantumOperator*)NULL);
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
            (HermitianQuantumOperator*)NULL, (HermitianQuantumOperator*)NULL);
    }
    ifs.close();

    HermitianQuantumOperator* observable_diag =
        new HermitianQuantumOperator(qubit_count);
    HermitianQuantumOperator* observable_non_diag =
        new HermitianQuantumOperator(qubit_count);

    for (UINT i = 0; i < ops.size(); ++i) {
        if (ops[i].find("X") != std::string::npos ||
            ops[i].find("Y") != std::string::npos) {
            observable_non_diag->add_operator(
                new PauliOperator(ops[i].c_str(), coefs[i]));
        } else {
            observable_diag->add_operator(
                new PauliOperator(ops[i].c_str(), coefs[i]));
        }
    }

    return std::make_pair(observable_diag, observable_non_diag);
}
}  // namespace observable
