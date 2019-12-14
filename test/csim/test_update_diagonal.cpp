
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

void test_single_diagonal_matrix_gate(std::function<void(UINT, const CTYPE*, CTYPE*, ITYPE)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	Eigen::MatrixXcd Identity(2, 2), Z(2, 2);
	Identity << 1, 0, 0, 1;
	Z << 1, 0, 0, -1;

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

	UINT target;
	double icoef, zcoef, norm;

	auto state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
	for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

	Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		// single qubit diagonal matrix gate
		target = rand_int(n);
		icoef = rand_real(); zcoef = rand_real();
		norm = sqrt(icoef * icoef + zcoef * zcoef);
		icoef /= norm; zcoef /= norm;
		U = icoef * Identity + 1.i*zcoef * Z;
		Eigen::VectorXcd diag = U.diagonal();
		func(target, (CTYPE*)diag.data(), state, dim);
		test_state = get_expanded_eigen_matrix_with_identity(target, U, n) * test_state;
		state_equal(state, test_state, dim, "single diagonal gate");
	}
	release_quantum_state(state);
}

TEST(UpdateTest, SingleDiagonalMatrixTest) {
	test_single_diagonal_matrix_gate(single_qubit_diagonal_matrix_gate);
	test_single_diagonal_matrix_gate(single_qubit_diagonal_matrix_gate_single_unroll);
#ifdef _OPENMP
	test_single_diagonal_matrix_gate(single_qubit_diagonal_matrix_gate_parallel_unroll);
#endif
#ifdef _USE_SIMD
	test_single_diagonal_matrix_gate(single_qubit_diagonal_matrix_gate_single_simd);
#ifdef _OPENMP
	test_single_diagonal_matrix_gate(single_qubit_diagonal_matrix_gate_parallel_simd);
#endif
#endif
}

void test_single_phase_gate(std::function<void(UINT, CTYPE, CTYPE*, ITYPE)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

	UINT target;
	double angle;

	auto state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
	for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

	Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		// single qubit phase matrix gate
		target = rand_int(n);
		angle = rand_real();
		U << 1, 0, 0, cos(angle) + 1.i*sin(angle);
		auto s = cos(angle) + 1.i*sin(angle);
		CTYPE t;
		__real__ t = s.real();
		__imag__ t = s.imag();
		func(target, t, state, dim);
		test_state = get_expanded_eigen_matrix_with_identity(target, U, n) * test_state;
		state_equal(state, test_state, dim, "single phase gate");
	}
	release_quantum_state(state);
}


TEST(UpdateTest, SinglePhaseGateTest) {
	test_single_phase_gate(single_qubit_phase_gate);
	test_single_phase_gate(single_qubit_phase_gate_single_unroll);
#ifdef _OPENMP
	test_single_phase_gate(single_qubit_phase_gate_parallel_unroll);
#endif
#ifdef _USE_SIMD
	test_single_phase_gate(single_qubit_phase_gate_single_simd);
#ifdef _OPENMP
	test_single_phase_gate(single_qubit_phase_gate_parallel_simd);
#endif
#endif
}
