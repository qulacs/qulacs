
#include <gtest/gtest.h>
#include "../util/util.h"
#include <Eigen/Core>
#include <string>
#include <algorithm>

#ifndef _MSC_VER
extern "C" {
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
#include <csim/init_ops.h>
}
#else
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
#include <csim/init_ops.h>
#endif
#include <csim/update_ops_cpp.hpp>

void test_single_dense_matrix_gate(std::function<void(UINT, const CTYPE*, CTYPE*, ITYPE)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

	UINT target;

	auto state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
	for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

	Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		// single qubit dense matrix gate
		// NOTE: Eigen uses column major by default. To use raw-data of eigen matrix, we need to specify RowMajor.
		target = rand_int(n);
		U = get_eigen_matrix_random_single_qubit_unitary();
		func(target, (CTYPE*)U.data(), state, dim);
		test_state = get_expanded_eigen_matrix_with_identity(target, U, n) * test_state;
		state_equal(state, test_state, dim, "single dense gate");
	}
	release_quantum_state(state);
}

TEST(UpdateTest, SingleDenseMatrixTest) {
	test_single_dense_matrix_gate(single_qubit_dense_matrix_gate);
	test_single_dense_matrix_gate(single_qubit_dense_matrix_gate_single_unroll);
#ifdef _OPENMP
	test_single_dense_matrix_gate(single_qubit_dense_matrix_gate_parallel_unroll);
#endif
#ifdef _USE_SIMD
	test_single_dense_matrix_gate(single_qubit_dense_matrix_gate_single_simd);
#ifdef _OPENMP
	test_single_dense_matrix_gate(single_qubit_dense_matrix_gate_parallel_simd);
#endif
#endif
}

void test_double_dense_matrix_gate(std::function<void(UINT, UINT, const CTYPE*, CTYPE*, ITYPE)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	std::vector<UINT> index_list;
	for (UINT i = 0; i < n; ++i) index_list.push_back(i);

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U, U2;
	Eigen::Matrix<std::complex<double>, 4, 4, Eigen::RowMajor> Umerge;

	UINT targets[2];

	auto state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
	for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

	Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

	for (UINT rep = 0; rep < max_repeat; ++rep) {

		// two qubit dense matrix gate
		U = get_eigen_matrix_random_single_qubit_unitary();
		U2 = get_eigen_matrix_random_single_qubit_unitary();

		std::random_shuffle(index_list.begin(), index_list.end());

		targets[0] = index_list[0];
		targets[1] = index_list[1];
		Umerge = kronecker_product(U2, U);
		// the below two lines are equivalent to the above two line
		//UINT targets_rev[2] = { targets[1], targets[0] };
		//Umerge = kronecker_product(U, U2);
		test_state = get_expanded_eigen_matrix_with_identity(targets[1], U2, n) * get_expanded_eigen_matrix_with_identity(targets[0], U, n) * test_state;
		func(targets[0], targets[1], (CTYPE*)Umerge.data(), state, dim);
		state_equal(state, test_state, dim, "two-qubit separable dense gate");
	}
	release_quantum_state(state);
}
TEST(UpdateTest, TwoQubitDenseMatrixTest) {
	test_double_dense_matrix_gate(double_qubit_dense_matrix_gate_c);
	test_double_dense_matrix_gate(double_qubit_dense_matrix_gate_single);
#ifdef _OPENMP
	test_double_dense_matrix_gate(double_qubit_dense_matrix_gate_parallel);
#endif
}


void test_general_dense_matrix_gate(std::function<void(const UINT*, UINT, const CTYPE*, CTYPE*, ITYPE)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	std::vector<UINT> index_list;
	for (UINT i = 0; i < n; ++i) index_list.push_back(i);

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U1, U2, U3;
	UINT targets[3];

	auto state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
	for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

	Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

	// general single
	{
		Eigen::Matrix<std::complex<double>, 2,2, Eigen::RowMajor> Umerge;
		for (UINT rep = 0; rep < max_repeat; ++rep) {

			// two qubit dense matrix gate
			U1 = get_eigen_matrix_random_single_qubit_unitary();
			std::random_shuffle(index_list.begin(), index_list.end());
			targets[0] = index_list[0];
			Umerge = U1;

			test_state = get_expanded_eigen_matrix_with_identity(targets[0], U1, n)
				* test_state;
			func(targets, 1, (CTYPE*)Umerge.data(), state, dim);
			state_equal(state, test_state, dim, "single-qubit separable dense gate");
		}
	}
	// general double
	{
		Eigen::Matrix<std::complex<double>, 4,4, Eigen::RowMajor> Umerge;

		for (UINT rep = 0; rep < max_repeat; ++rep) {

			// two qubit dense matrix gate
			U1 = get_eigen_matrix_random_single_qubit_unitary();
			U2 = get_eigen_matrix_random_single_qubit_unitary();

			std::random_shuffle(index_list.begin(), index_list.end());
			targets[0] = index_list[0];
			targets[1] = index_list[1];
			Umerge = kronecker_product(U2, U1);

			test_state =
				get_expanded_eigen_matrix_with_identity(targets[1], U2, n)
				* get_expanded_eigen_matrix_with_identity(targets[0], U1, n)
				* test_state;
			func(targets, 2, (CTYPE*)Umerge.data(), state, dim);
			state_equal(state, test_state, dim, "two-qubit separable dense gate");
		}
	}
	// general triple
	{
		Eigen::Matrix<std::complex<double>, 8, 8, Eigen::RowMajor> Umerge;

		for (UINT rep = 0; rep < max_repeat; ++rep) {

			// two qubit dense matrix gate
			U1 = get_eigen_matrix_random_single_qubit_unitary();
			U2 = get_eigen_matrix_random_single_qubit_unitary();
			U3 = get_eigen_matrix_random_single_qubit_unitary();

			std::random_shuffle(index_list.begin(), index_list.end());
			targets[0] = index_list[0];
			targets[1] = index_list[1];
			targets[2] = index_list[2];
			Umerge = kronecker_product(U3, kronecker_product(U2, U1));

			test_state =
				get_expanded_eigen_matrix_with_identity(targets[2], U3, n)
				* get_expanded_eigen_matrix_with_identity(targets[1], U2, n)
				* get_expanded_eigen_matrix_with_identity(targets[0], U1, n)
				* test_state;
			func(targets, 3, (CTYPE*)Umerge.data(), state, dim);
			state_equal(state, test_state, dim, "three-qubit separable dense gate");
		}
	}
	release_quantum_state(state);
}

TEST(UpdateTest, ThreeQubitDenseMatrixTest) {
	test_general_dense_matrix_gate(multi_qubit_dense_matrix_gate);
	test_general_dense_matrix_gate(multi_qubit_dense_matrix_gate_single);
#ifdef _OPENMP
	test_general_dense_matrix_gate(multi_qubit_dense_matrix_gate_parallel);
#endif
}