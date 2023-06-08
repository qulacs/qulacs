#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <cppsim/circuit.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_named_pauli.hpp>
#include <cppsim/gate_to_gqo.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/state_dm.hpp>
#include <cppsim/type.hpp>
#include <cppsim/utility.hpp>
#include <csim/constant.hpp>
#include <csim/update_ops.hpp>
#include <fstream>
#include <functional>

#include "../util/util.hpp"

TEST(ObservableTest, CheckExpectationValue) {
    const UINT n = 4;
    const UINT dim = 1ULL << n;

    double coef;
    CPPCTYPE res;
    CPPCTYPE test_res;
    Random random;

    const auto X = make_X();

    QuantumState state(n);
    state.set_computational_basis(0);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    test_state(0) = 1.;

    coef = random.uniform();
    Observable observable(n);
    observable.add_operator(coef, "X 0");
    Eigen::MatrixXcd test_observable = Eigen::MatrixXcd::Zero(dim, dim);
    test_observable += coef * get_expanded_eigen_matrix_with_identity(0, X, n);

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state)(0, 0);
    ASSERT_NEAR(test_res.real(), res.real(), eps);
    ASSERT_NEAR(res.imag(), 0, eps);
    ASSERT_NEAR(test_res.imag(), 0, eps);

    state.set_Haar_random_state();
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state.data_cpp()[i];
    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state)(0, 0);
    ASSERT_NEAR(test_res.real(), res.real(), eps);
    ASSERT_NEAR(res.imag(), 0, eps);
    ASSERT_NEAR(test_res.imag(), 0, eps);

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        Observable rand_observable(n);
        Eigen::MatrixXcd test_rand_observable =
            Eigen::MatrixXcd::Zero(dim, dim);

        UINT term_count = random.int32() % 10 + 1;
        for (UINT term = 0; term < term_count; ++term) {
            std::vector<UINT> paulis(n, 0);
            Eigen::MatrixXcd test_rand_observable_term =
                Eigen::MatrixXcd::Identity(dim, dim);
            coef = random.uniform();
            for (UINT i = 0; i < paulis.size(); ++i) {
                paulis[i] = random.int32() % 4;

                test_rand_observable_term *=
                    get_expanded_eigen_matrix_with_identity(
                        i, get_eigen_matrix_single_Pauli(paulis[i]), n);
            }
            test_rand_observable += coef * test_rand_observable_term;

            std::string str = "";
            for (UINT ind = 0; ind < paulis.size(); ind++) {
                UINT val = paulis[ind];
                if (val != 0) {
                    if (val == 1)
                        str += " X";
                    else if (val == 2)
                        str += " Y";
                    else if (val == 3)
                        str += " Z";
                    str += " " + std::to_string(ind);
                }
            }
            rand_observable.add_operator(coef, str.c_str());
        }

        state.set_Haar_random_state();
        for (ITYPE i = 0; i < dim; ++i) test_state[i] = state.data_cpp()[i];

        res = rand_observable.get_expectation_value(&state);
        test_res =
            (test_state.adjoint() * test_rand_observable * test_state)(0, 0);
        ASSERT_NEAR(test_res.real(), res.real(), eps);
        ASSERT_NEAR(res.imag(), 0, eps);
        ASSERT_NEAR(test_res.imag(), 0, eps);
    }
}

TEST(ObservableTest, CheckParsedObservableFromOpenFermionText) {
    auto func = [](const std::string str,
                    const QuantumStateBase* state) -> CPPCTYPE {
        CPPCTYPE energy = 0;

        std::vector<std::string> lines = split(str, "\n");

        for (std::string line : lines) {
            // std::cout << state->get_norm() << std::endl;

            std::vector<std::string> elems;
            elems = split(line, "()j[]+");

            chfmt(elems[3]);

            CPPCTYPE coef(std::stod(elems[0]), std::stod(elems[1]));
            // std::cout << elems[3].c_str() << std::endl;

            PauliOperator mpt(elems[3].c_str(), coef.real());

            // std::cout << mpt.get_coef() << " ";
            // std::cout << elems[3].c_str() << std::endl;
            energy += mpt.get_expectation_value(state);
            // mpt.get_expectation_value(state);
        }
        return energy;
    };

    const std::string text =
        "(-0.8126100000000005+0j) [] +\n"
        "(0.04532175+0j) [X0 Z1 X2] +\n"
        "(0.04532175+0j) [X0 Z1 X2 Z3] +\n"
        "(0.04532175+0j) [Y0 Z1 Y2] +\n"
        "(0.04532175+0j) [Y0 Z1 Y2 Z3] +\n"
        "(0.17120100000000002+0j) [Z0] +\n"
        "(0.17120100000000002+0j) [Z0 Z1] +\n"
        "(0.165868+0j) [Z0 Z1 Z2] +\n"
        "(0.165868+0j) [Z0 Z1 Z2 Z3] +\n"
        "(0.12054625+0j) [Z0 Z2] +\n"
        "(0.12054625+0j) [Z0 Z2 Z3] +\n"
        "(0.16862325+0j) [Z1] +\n"
        "(-0.22279649999999998+0j) [Z1 Z2 Z3] +\n"
        "(0.17434925+0j) [Z1 Z3] +\n"
        "(-0.22279649999999998+0j) [Z2]";

    CPPCTYPE res, test_res;

    Observable* observable;
    observable = observable::create_observable_from_openfermion_text(text);
    ASSERT_NE(observable, (Observable*)NULL);
    UINT qubit_count = observable->get_qubit_count();

    QuantumState state(qubit_count);
    state.set_computational_basis(0);

    res = observable->get_expectation_value(&state);
    test_res = func(text, &state);

    ASSERT_NEAR(test_res.real(), res.real(), eps);
    ASSERT_NEAR(test_res.imag(), res.imag(), eps);

    state.set_Haar_random_state();

    res = observable->get_expectation_value(&state);
    test_res = func(text, &state);

    ASSERT_NEAR(test_res.real(), res.real(), eps);
    ASSERT_NEAR(test_res.imag(), 0, eps);
    ASSERT_NEAR(res.imag(), 0, eps);

    delete observable;
}

TEST(ObservableTest, CheckGetMatrix) {
    const std::string text =
        "(-0.8126100000000005+0j) [] +\n"
        "(0.04532175+0j) [X0 Z1 X2] +\n"
        "(0.04532175+0j) [X0 Z1 X2 Z3] +\n"
        "(0.04532175+0j) [Y0 Z1 Y2] +\n"
        "(0.04532175+0j) [Y0 Z1 Y2 Z3] +\n"
        "(0.17120100000000002+0j) [Z0] +\n"
        "(0.17120100000000002+0j) [Z0 Z1] +\n"
        "(0.165868+0j) [Z0 Z1 Z2] +\n"
        "(0.165868+0j) [Z0 Z1 Z2 Z3] +\n"
        "(0.12054625+0j) [Z0 Z2] +\n"
        "(0.12054625+0j) [Z0 Z2 Z3] +\n"
        "(0.16862325+0j) [Z1] +\n"
        "(-0.22279649999999998+0j) [Z1 Z2 Z3] +\n"
        "(0.17434925+0j) [Z1 Z3] +\n"
        "(-0.22279649999999998+0j) [Z2]";

    Observable* observable;
    observable = observable::create_observable_from_openfermion_text(text);
    ASSERT_NE(observable, (Observable*)NULL);
    UINT qubit_count = observable->get_qubit_count();

    SparseComplexMatrixRowMajor result = observable->get_matrix(),
                                expected(16, 16);

    expected.insert(0, 0) = CPPCTYPE(-4.163336342344337e-16, 0.0);
    expected.insert(1, 1) = CPPCTYPE(-1.8304610000000006, 0.0);
    expected.insert(1, 4) = CPPCTYPE(0.181287, 0.0);
    expected.insert(2, 2) = CPPCTYPE(-1.2462260000000005, 0.0);
    expected.insert(3, 3) = CPPCTYPE(-1.0649390000000005, 0.0);
    expected.insert(3, 6) = CPPCTYPE(-0.181287, 0.0);
    expected.insert(4, 1) = CPPCTYPE(0.181287, 0.0);
    expected.insert(4, 4) = CPPCTYPE(-0.25447100000000045, 0.0);
    expected.insert(5, 5) = CPPCTYPE(0.20638199999999932, 0.0);
    expected.insert(6, 3) = CPPCTYPE(-0.181287, 0.0);
    expected.insert(6, 6) = CPPCTYPE(-1.0649390000000005, 0.0);
    expected.insert(7, 7) = CPPCTYPE(-1.2462260000000005, 0.0);
    expected.insert(8, 8) = CPPCTYPE(-0.4759340000000004, 0.0);
    expected.insert(9, 9) = CPPCTYPE(-1.1607380000000005, 0.0);
    expected.insert(10, 10) = CPPCTYPE(-1.2524770000000005, 0.0);
    expected.insert(11, 11) = CPPCTYPE(-1.2524770000000005, 0.0);
    expected.insert(12, 12) = CPPCTYPE(-0.4759340000000005, 0.0);
    expected.insert(13, 13) = CPPCTYPE(-1.1607380000000007, 0.0);
    expected.insert(14, 14) = CPPCTYPE(-0.3612910000000006, 0.0);
    expected.insert(15, 15) = CPPCTYPE(-0.3612910000000006, 0.0);

    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            ASSERT_NEAR(result.coeffRef(i, j).real(),
                expected.coeffRef(i, j).real(), eps);
            ASSERT_NEAR(result.coeffRef(i, j).imag(),
                expected.coeffRef(i, j).imag(), eps);
        }
    }

    delete observable;
}

/*

TEST(ObservableTest, CheckParsedObservableFromOpenFermionFile) {
    auto func = [](const std::string path,
        const QuantumStateBase* state) -> CPPCTYPE {
        std::ifstream ifs;
        ifs.open(path);
        if (!ifs) {
            std::cerr << "ERROR: Cannot open file" << std::endl;
            return -1.;
        }

        CPPCTYPE energy = 0;

        std::string str;
        while (getline(ifs, str)) {
            // std::cout << state->get_norm() << std::endl;

            std::vector<std::string> elems;
            elems = split(str, "()j[]+");

            chfmt(elems[3]);

            CPPCTYPE coef(std::stod(elems[0]), std::stod(elems[1]));
            // std::cout << elems[3].c_str() << std::endl;

            PauliOperator mpt(elems[3].c_str(), coef.real());

            // std::cout << mpt.get_coef() << " ";
            // std::cout << elems[3].c_str() << std::endl;
            energy += mpt.get_expectation_value(state);
            // mpt.get_expectation_value(state);
        }
        if (!ifs.eof()) {
            std::cerr << "ERROR: Invalid format" << std::endl;
            return -1.;
        }
        ifs.close();
        return energy;
    };


    const char* filename = "../test/cppsim/H2.txt";

    CPPCTYPE res, test_res;

    Observable* observable;
    observable = observable::create_observable_from_openfermion_file(filename);
    ASSERT_NE(observable, (Observable*)NULL);
    UINT qubit_count = observable->get_qubit_count();

    QuantumState state(qubit_count);
    state.set_computational_basis(0);

    res = observable->get_expectation_value(&state);
    test_res = func(filename, &state);

    ASSERT_EQ(test_res, res);

    state.set_Haar_random_state();

    res = observable->get_expectation_value(&state);
    test_res = func(filename, &state);

    ASSERT_NEAR(test_res.real(), res.real(), eps);
    ASSERT_NEAR(test_res.imag(), 0, eps);
    ASSERT_NEAR(res.imag(), 0, eps);
}

TEST(ObservableTest, CheckSplitObservable) {
    auto func = [](const std::string path,
                    const QuantumStateBase* state) -> CPPCTYPE {
        std::ifstream ifs;
        CPPCTYPE coef;
        ifs.open(path);
        if (!ifs) {
            std::cerr << "ERROR: Cannot open file" << std::endl;
            return -1.;
        }

        CPPCTYPE energy = 0;

        std::string str;
        while (getline(ifs, str)) {
            // std::cout << state->get_norm() << std::endl;

            std::vector<std::string> elems;
            elems = split(str, "()j[]+");

            chfmt(elems[3]);

            CPPCTYPE coef(std::stod(elems[0]), std::stod(elems[1]));
            // std::cout << elems[3].c_str() << std::endl;

            PauliOperator mpt(elems[3].c_str(), coef.real());

            // std::cout << mpt.get_coef() << " ";
            // std::cout << elems[3].c_str() << std::endl;
            energy += mpt.get_expectation_value(state);
            // mpt.get_expectation_value(state);
        }
        if (!ifs.eof()) {
            std::cerr << "ERROR: Invalid format" << std::endl;
            return -1.;
        }
        ifs.close();
        return energy;
    };


    const char* filename = "../test/cppsim/H2.txt";

    CPPCTYPE diag_res, test_res, non_diag_res;

    std::pair<Observable*, Observable*> observables;
    observables = observable::create_split_observable(filename);
    ASSERT_NE(observables.first, (Observable*)NULL);
    ASSERT_NE(observables.second, (Observable*)NULL);

    UINT qubit_count = observables.first->get_qubit_count();
    QuantumState state(qubit_count);
    state.set_computational_basis(0);

    diag_res = observables.first->get_expectation_value(&state);
    non_diag_res = observables.second->get_expectation_value(&state);
    test_res = func(filename, &state);

    ASSERT_NEAR(test_res.real(), (diag_res + non_diag_res).real(), eps);
    ASSERT_NEAR(test_res.imag(), 0, eps);
    ASSERT_NEAR(diag_res.imag(), 0, eps);
    ASSERT_NEAR(non_diag_res.imag(), 0, eps);

    state.set_Haar_random_state();

    diag_res = observables.first->get_expectation_value(&state);
    non_diag_res = observables.second->get_expectation_value(&state);
    test_res = func(filename, &state);

    ASSERT_NEAR(test_res.real(), (diag_res + non_diag_res).real(), eps);
    ASSERT_NEAR(test_res.imag(), 0, eps);
    ASSERT_NEAR(diag_res.imag(), 0, eps);
    ASSERT_NEAR(non_diag_res.imag(), 0, eps);
}

*/

// Kind of eigenvalue calculation method.
// Only used to specify method in `test_eigenvalue()`.
enum class CalculationMethod {
    PowerMethod,
    ArnoldiMethod,
    LanczosMethod,
};

// Test calculating eigenvalue.
// Actual test code calls this function with prepared observable.
// Return an error message if failed, an empty string if passed.
std::string test_eigenvalue(Observable& observable, const UINT iter_count,
    const double eps, const CalculationMethod method) {
    auto observable_matrix = convert_observable_to_matrix(observable);
    const auto eigenvalues = observable_matrix.eigenvalues();
    CPPCTYPE test_ground_state_eigenvalue = eigenvalues[0];
    for (UINT i = 0; i < eigenvalues.size(); i++) {
        if (eigenvalues[i].real() < test_ground_state_eigenvalue.real()) {
            test_ground_state_eigenvalue = eigenvalues[i];
        }
    }

    const auto qubit_count = observable.get_qubit_count();
    QuantumState state(qubit_count);
    state.set_Haar_random_state();
    CPPCTYPE ground_state_eigenvalue;
    if (method == CalculationMethod::PowerMethod) {
        ground_state_eigenvalue =
            observable.solve_ground_state_eigenvalue_by_power_method(
                &state, iter_count);
    } else if (method == CalculationMethod::ArnoldiMethod) {
        ground_state_eigenvalue =
            observable.solve_ground_state_eigenvalue_by_arnoldi_method(
                &state, iter_count);
    } else if (method == CalculationMethod::LanczosMethod) {
        ground_state_eigenvalue =
            observable.solve_ground_state_eigenvalue_by_lanczos_method(
                &state, iter_count);
    }
    std::string err_message;
    err_message = _CHECK_NEAR(ground_state_eigenvalue.real(),
        test_ground_state_eigenvalue.real(), eps);
    if (err_message != "") return err_message;

    QuantumState multiplied_state(qubit_count);
    QuantumState work_state(qubit_count);
    // multiplied_state = A|q>
    observable.apply_to_state(&work_state, state, &multiplied_state);
    // state = λ|q>
    state.multiply_coef(ground_state_eigenvalue);
    multiplied_state.normalize(multiplied_state.get_squared_norm());
    state.normalize(state.get_squared_norm());

    for (UINT i = 0; i < state.dim; i++) {
        err_message = _CHECK_NEAR(multiplied_state.data_cpp()[i].real(),
            state.data_cpp()[i].real(), eps);
        if (err_message != "") return err_message;
        err_message = _CHECK_NEAR(multiplied_state.data_cpp()[i].imag(),
            state.data_cpp()[i].imag(), eps);
        if (err_message != "") return err_message;
    }
    return "";
}

TEST(ObservableTest, MinimumEigenvalueByPowerMethod) {
    const double in_eps = 1e-2;
    const UINT qubit_count = 4;
    const UINT test_count = 10;
    UINT pass_count = 0;
    Random random;

    for (UINT i = 0; i < test_count; i++) {
        const UINT operator_count =
            random.int32() % 10 + 2;  // 2 <= operator_count <= 11
        auto observable = Observable(qubit_count);
        observable.add_random_operator(operator_count);
        std::string err_message = test_eigenvalue(
            observable, 500, in_eps, CalculationMethod::PowerMethod);
        if (err_message == "")
            pass_count++;
        else
            std::cerr << err_message;
    }
    ASSERT_GE(pass_count, test_count - 1);
}

TEST(ObservableTest, MinimumEigenvalueByArnoldiMethod) {
    const double in_eps = 1e-6;
    const UINT test_count = 10;
    UINT pass_count = 0;
    Random random;

    for (UINT i = 0; i < test_count; i++) {
        // 3 <= qubit_count <= 5
        const auto qubit_count = random.int32() % 4 + 3;
        // 2 <= operator_count <= 11
        const auto operator_count = random.int32() % 10 + 2;
        auto observable = Observable(qubit_count);
        observable.add_random_operator(operator_count);
        std::string err_message = test_eigenvalue(
            observable, 60, in_eps, CalculationMethod::ArnoldiMethod);
        if (err_message == "")
            pass_count++;
        else
            std::cerr << err_message;
    }
    ASSERT_GE(pass_count, test_count - 1);
}

void add_identity(Observable* observable, Random random) {
    std::vector<UINT> qil;
    std::vector<UINT> qpl;
    for (UINT i = 0; i < observable->get_qubit_count(); i++) {
        qil.push_back(i);
        qpl.push_back(0);
    }
    auto op = new PauliOperator(qil, qpl, random.uniform());
    observable->add_operator_move(op);
}

// Test observable with identity pauli operator because calculation was unstable
// in this situation.
TEST(ObservableTest, MinimumEigenvalueByArnoldiMethodWithIdentity) {
    const double in_eps = 1e-6;
    const UINT test_count = 10;
    UINT pass_count = 0;
    Random random;

    for (UINT i = 0; i < test_count; i++) {
        // 3 <= qubit_count <= 5
        const auto qubit_count = random.int32() % 4 + 3;
        // 2 <= operator_count <= 11
        const auto operator_count = random.int32() % 10 + 2;
        auto observable = Observable(qubit_count);
        observable.add_random_operator(operator_count);
        add_identity(&observable, random);
        std::string err_message = test_eigenvalue(
            observable, 70, in_eps, CalculationMethod::ArnoldiMethod);
        if (err_message == "")
            pass_count++;
        else
            std::cerr << err_message;
    }
    ASSERT_GE(pass_count, test_count - 1);
}

TEST(ObservableTest, MinimumEigenvalueByLanczosMethod) {
    const double in_eps = 1e-6;
    const UINT test_count = 10;
    UINT pass_count = 0;
    Random random;

    for (UINT i = 0; i < test_count; i++) {
        // 3 <= qubit_count <= 6
        const auto qubit_count = random.int32() % 4 + 3;
        // 2 <= operator_count <= 11
        const auto operator_count = random.int32() % 10 + 2;
        auto observable = Observable(qubit_count);
        observable.add_random_operator(operator_count);
        std::string err_message = test_eigenvalue(
            observable, 70, in_eps, CalculationMethod::LanczosMethod);
        if (err_message == "")
            pass_count++;
        else
            std::cerr << err_message;
    }
    ASSERT_GE(pass_count, test_count - 1);
}

TEST(ObservableTest, GetDaggerTest) {
    const UINT qubit_count = 4;

    auto observable = Observable(qubit_count);
    observable.add_operator(1.0, "X 0");
    auto dagger_observable = observable.get_dagger();
    std::string s = dagger_observable->to_string();
    ASSERT_TRUE(s == "(1,-0) X 0" || s == "(1,0) X 0");
    delete dagger_observable;
}

TEST(ObservableTest, ObservableAndStateHaveDifferentQubitCountTest) {
    auto func = [](const std::string str,
                    const QuantumStateBase* state) -> CPPCTYPE {
        CPPCTYPE energy = 0;

        std::vector<std::string> lines = split(str, "\n");

        for (std::string line : lines) {
            std::vector<std::string> elems;
            elems = split(line, "()j[]+");
            chfmt(elems[3]);
            CPPCTYPE coef(std::stod(elems[0]), std::stod(elems[1]));
            PauliOperator mpt(elems[3].c_str(), coef.real());
            energy += mpt.get_expectation_value(state);
        }
        return energy;
    };

    const std::string text =
        "(-0.8126100000000005+0j) [] +\n"
        "(0.04532175+0j) [X0 Z1 X2] +\n"
        "(0.04532175+0j) [X0 Z1 X2 Z3] +\n"
        "(0.04532175+0j) [Y0 Z1 Y2] +\n"
        "(0.04532175+0j) [Y0 Z1 Y2 Z3] +\n"
        "(0.17120100000000002+0j) [Z0] +\n"
        "(0.17120100000000002+0j) [Z0 Z1] +\n"
        "(0.165868+0j) [Z0 Z1 Z2] +\n"
        "(0.165868+0j) [Z0 Z1 Z2 Z3] +\n"
        "(0.12054625+0j) [Z0 Z2] +\n"
        "(0.12054625+0j) [Z0 Z2 Z3] +\n"
        "(0.16862325+0j) [Z1] +\n"
        "(-0.22279649999999998+0j) [Z1 Z2 Z3] +\n"
        "(0.17434925+0j) [Z1 Z3] +\n"
        "(-0.22279649999999998+0j) [Z2]";

    CPPCTYPE res, test_res;

    Observable* observable;
    observable = observable::create_observable_from_openfermion_text(text);
    ASSERT_NE(observable, (Observable*)NULL);
    UINT qubit_count = observable->get_qubit_count();

    QuantumState state(qubit_count + 2);  // +2 is point. diff test
    state.set_computational_basis(0);

    res = observable->get_expectation_value(&state);
    test_res = func(text, &state);

    ASSERT_NEAR(res.real(), test_res.real(), eps);
    ASSERT_NEAR(res.imag(), test_res.imag(), eps);

    state.set_Haar_random_state();

    res = observable->get_expectation_value(&state);
    test_res = func(text, &state);

    ASSERT_NEAR(res.real(), test_res.real(), eps);
    ASSERT_NEAR(0, test_res.imag(), eps);
    ASSERT_NEAR(0, res.imag(), eps);

    delete observable;
}

TEST(ObservableTest, ApplyIdentityToState) {
    double coef = .5;
    int n_qubits = 3;
    Observable obs(n_qubits);
    obs.add_operator(coef, "I 0");
    QuantumState state(n_qubits);
    QuantumState dst_state(n_qubits);
    obs.apply_to_state(&state, &dst_state);
    state.add_state_with_coef(-1 / coef, &dst_state);
    ASSERT_NEAR(0., state.get_squared_norm(), eps);
}

TEST(gate_to_general_quantum_operatorTest, Random4bit) {
    QuantumGateBase* random_gate = gate::RandomUnitary({0, 1, 2, 3});

    auto GQO_ret = to_general_quantum_operator(random_gate, 4);
    QuantumState stateA(4);
    stateA.set_Haar_random_state();
    QuantumState stateB(4);
    GQO_ret->apply_to_state(&stateA, &stateB);
    random_gate->update_quantum_state(&stateA);

    double inpro = state::inner_product(&stateA, &stateB).real();
    ASSERT_NEAR(inpro, 1.0, 0.001);

    delete random_gate;
    delete GQO_ret;
}

TEST(gate_to_general_quantum_operatorTest, Random1bit) {
    QuantumGateBase* random_gate = gate::RandomUnitary({0});

    auto GQO_ret = to_general_quantum_operator(random_gate, 1);
    QuantumState stateA(1);
    stateA.set_Haar_random_state();
    QuantumState stateB(1);
    GQO_ret->apply_to_state(&stateA, &stateB);
    random_gate->update_quantum_state(&stateA);

    double inpro = state::inner_product(&stateA, &stateB).real();

    ASSERT_NEAR(inpro, 1.0, 0.001);
    delete random_gate;
    delete GQO_ret;
}

TEST(gate_to_general_quantum_operatorTest, XYgate) {
    auto gate_X = gate::X(0);
    auto gate_Y = gate::Y(1);
    QuantumGateBase* random_gate = gate::merge(gate_X, gate_Y);
    delete gate_X;
    delete gate_Y;

    auto GQO_ret = to_general_quantum_operator(random_gate, 2);
    QuantumState stateA(2);
    stateA.set_Haar_random_state();
    QuantumState stateB(2);
    GQO_ret->apply_to_state(&stateA, &stateB);
    random_gate->update_quantum_state(&stateA);

    double inpro = state::inner_product(&stateA, &stateB).real();
    ASSERT_NEAR(inpro, 1.0, 0.001);

    delete random_gate;
    delete GQO_ret;
}
