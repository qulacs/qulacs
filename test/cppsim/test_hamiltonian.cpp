#include <gtest/gtest.h>
#include <csim/constant.h>
#include <cppsim/type.hpp>
#include "../util/util.h"
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/hamiltonian.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/utility.hpp>
#include <fstream>



TEST(HamiltonianTest, CheckExpectationValue) {
	const UINT n = 4;
	const UINT dim = 1ULL << n;
	const double eps = 1e-14;
	double coef;
	double res;
	std::complex<double> test_res;
	Random random;

	Eigen::MatrixXcd X(2, 2);
	X << 0, 1, 1, 0;

	QuantumState state(n);
	state.set_computational_basis(0);
	Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
	test_state(0) = 1.;

	coef = random.uniform();
	Hamiltonian ham(n);
	ham.add_operator(coef, "X 0");
	Eigen::MatrixXcd test_ham = Eigen::MatrixXcd::Zero(dim, dim);
	test_ham += coef*get_expanded_eigen_matrix_with_identity(0, X, n);

	res = ham.get_expectation_value(&state);
	test_res = (test_state.adjoint() * test_ham * test_state);
	ASSERT_NEAR(test_res.real(), res, eps);
	ASSERT_NEAR(test_res.imag(), 0, eps);

	state.set_Haar_random_state();
	for (ITYPE i = 0; i < dim; ++i) test_state[i] = state.data_cpp()[i];
	res = ham.get_expectation_value(&state);
	test_res = (test_state.adjoint() * test_ham * test_state);
	ASSERT_NEAR(test_res.real(), res, eps);
	ASSERT_NEAR(test_res.imag(), 0, eps);



	for (UINT repeat = 0; repeat < 10; ++repeat) {

		Hamiltonian rand_ham(n);
		Eigen::MatrixXcd test_rand_ham = Eigen::MatrixXcd::Zero(dim, dim);

		UINT term_count = random.int32() % 10+1;
		for (UINT term = 0; term < term_count; ++term) {
			std::vector<UINT> paulis(n,0);
			Eigen::MatrixXcd test_rand_ham_term = Eigen::MatrixXcd::Identity(dim, dim);
			coef = random.uniform();
			for (UINT i = 0; i < paulis.size(); ++i) {
				paulis[i] = random.int32() % 4;

				test_rand_ham_term *= get_expanded_eigen_matrix_with_identity(i, get_eigen_matrix_single_Pauli(paulis[i]) , n);
			}
			test_rand_ham += coef* test_rand_ham_term;

			std::string str = "";
			for (UINT ind = 0; ind < paulis.size(); ind++) {
				UINT val = paulis[ind];
				if (val != 0) {
					if (val == 1) str += " X";
					else if (val == 2) str += " Y";
					else if (val == 3) str += " Z";
					str += " " + std::to_string(ind);
				}
			}
			rand_ham.add_operator(coef, str.c_str());
		}

		state.set_Haar_random_state();
		for (ITYPE i = 0; i < dim; ++i) test_state[i] = state.data_cpp()[i];

		res = rand_ham.get_expectation_value(&state);
		test_res = test_state.adjoint() * test_rand_ham * test_state;
		ASSERT_NEAR(test_res.real(), res, eps);
		ASSERT_NEAR(test_res.imag(), 0, eps);

	}
}

TEST(HamiltonianTest, CheckParsedHamiltonian){
	auto func = [](const std::string path, const QuantumStateBase* state) -> double {
		    std::ifstream ifs;
			ifs.open(path);
			if (!ifs){
				std::cerr << "ERROR: Cannot open file" << std::endl;
				std::exit(EXIT_FAILURE);
			}

			double energy = 0;

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
			}
			ifs.close();
			return energy;
	};

	const double eps = 1e-14;
	const char* filename = "../test/cppsim/H2.txt";

	double res, test_res;


	Hamiltonian ham(filename);
    UINT qubit_count = ham.get_qubit_count();

	QuantumState state(qubit_count);
	state.set_computational_basis(0);

	res = ham.get_expectation_value(&state);
	test_res = func(filename, &state);

	ASSERT_EQ(test_res, res);


	state.set_Haar_random_state();

	res = ham.get_expectation_value(&state);
	test_res = func(filename, &state);

	ASSERT_NEAR(test_res, res, eps);
}

TEST(HamiltonianTest, CheckSplitHamiltonian){
	auto func = [](const std::string path, const QuantumStateBase* state) -> double {
		    std::ifstream ifs;
			CPPCTYPE coef;
			ifs.open(path);
			if (!ifs){
				std::cerr << "ERROR: Cannot open file" << std::endl;
				std::exit(EXIT_FAILURE);
			}

			double energy = 0;

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
			}
			ifs.close();
			return energy;
	};

	const double eps = 1e-14;
	const char* filename  = "../test/cppsim/H2.txt";

	double diag_res, test_res, non_diag_res;

    std::pair<Hamiltonian*, Hamiltonian*> hams = Hamiltonian::get_split_hamiltonian(filename);

    UINT qubit_count = hams.first->get_qubit_count();
    QuantumState state(qubit_count);
	state.set_computational_basis(0);


	diag_res = hams.first->get_expectation_value(&state);
	non_diag_res = hams.second->get_expectation_value(&state);
	test_res = func(filename, &state);

	ASSERT_NEAR(test_res, diag_res + non_diag_res, eps);


	state.set_Haar_random_state();

	diag_res = hams.first->get_expectation_value(&state);
	non_diag_res = hams.second->get_expectation_value(&state);
	test_res = func(filename, &state);

	ASSERT_NEAR(test_res, diag_res + non_diag_res, eps);

}

