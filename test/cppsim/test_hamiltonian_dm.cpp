#include <gtest/gtest.h>
#include <csim/constant.h>
#include <cppsim/type.hpp>
#include "../util/util.h"
#include <cppsim/state.hpp>
#include <cppsim/state_dm.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/utility.hpp>
#include <fstream>



TEST(DensityMatrixObservableTest, CheckExpectationValue) {
	const UINT n = 4;
	const UINT dim = 1ULL << n;
	const double eps = 1e-14;
	double coef;
	CPPCTYPE res_vec,res_mat;
	CPPCTYPE test_res;
	Random random;

	Eigen::MatrixXcd X(2, 2);
	X << 0, 1, 1, 0;

	QuantumState vector_state(n);
	DensityMatrix density_matrix(n);
	vector_state.set_computational_basis(0);
	density_matrix.load(&vector_state);

	coef = random.uniform();
	Observable observable(n);
	observable.add_operator(coef, "X 0");

	res_vec = observable.get_expectation_value(&vector_state);
	res_mat = observable.get_expectation_value(&density_matrix);
	ASSERT_NEAR(res_vec.real(), res_mat.real(), eps);
	ASSERT_NEAR(res_vec.imag(), 0, eps);
	ASSERT_NEAR(res_mat.imag(), 0, eps);

	vector_state.set_Haar_random_state();
	density_matrix.load(&vector_state);
	res_vec = observable.get_expectation_value(&vector_state);
	res_mat = observable.get_expectation_value(&density_matrix);
	ASSERT_NEAR(res_vec.real(), res_mat.real(), eps);
	ASSERT_NEAR(res_vec.imag(), 0, eps);
	ASSERT_NEAR(res_mat.imag(), 0, eps);

	for (UINT repeat = 0; repeat < 10; ++repeat) {

		Observable rand_observable(n);
		Eigen::MatrixXcd test_rand_observable = Eigen::MatrixXcd::Zero(dim, dim);

		UINT term_count = random.int32() % 10 + 1;
		for (UINT term = 0; term < term_count; ++term) {
			std::vector<UINT> paulis(n, 0);
			Eigen::MatrixXcd test_rand_observable_term = Eigen::MatrixXcd::Identity(dim, dim);
			coef = random.uniform();
			for (UINT i = 0; i < paulis.size(); ++i) {
				paulis[i] = random.int32() % 4;

				test_rand_observable_term *= get_expanded_eigen_matrix_with_identity(i, get_eigen_matrix_single_Pauli(paulis[i]), n);
			}
			test_rand_observable += coef * test_rand_observable_term;

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
			rand_observable.add_operator(coef, str.c_str());
		}

		vector_state.set_Haar_random_state();
		density_matrix.load(&vector_state);

		res_vec = observable.get_expectation_value(&vector_state);
		res_mat = observable.get_expectation_value(&density_matrix);
		ASSERT_NEAR(res_vec.real(), res_mat.real(), eps);
		ASSERT_NEAR(res_vec.imag(), 0, eps);
		ASSERT_NEAR(res_mat.imag(), 0, eps);
	}
}


TEST(DensityMatrixObservableTest, CheckParsedObservableFromOpenFermionText) {
	auto func = [](const std::string str, const QuantumStateBase* state) -> CPPCTYPE {
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

	const double eps = 1e-14;
	const std::string text = "(-0.8126100000000005+0j) [] +\n"
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

	CPPCTYPE res_vec, res_mat;

	Observable* observable;
	observable = observable::create_observable_from_openfermion_text(text);
	ASSERT_NE(observable, (Observable*)NULL);
	UINT qubit_count = observable->get_qubit_count();

	QuantumState vector_state(qubit_count);
	DensityMatrix density_matrix(qubit_count);

	vector_state.set_computational_basis(0);
	density_matrix.load(&vector_state);
	res_vec = observable->get_expectation_value(&vector_state);
	res_mat = observable->get_expectation_value(&density_matrix);
	ASSERT_NEAR(res_vec.real(), res_mat.real(), eps);
	ASSERT_NEAR(res_vec.imag(), res_mat.imag(), eps);


	vector_state.set_Haar_random_state();
	density_matrix.load(&vector_state);
	res_vec = observable->get_expectation_value(&vector_state);
	res_mat = observable->get_expectation_value(&density_matrix);
	ASSERT_NEAR(res_vec.real(), res_mat.real(), eps);
	ASSERT_NEAR(res_vec.imag(), res_mat.imag(), eps);
}
