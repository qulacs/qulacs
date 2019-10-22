
#include "test_util.hpp"
#include <gpusim/memory_ops.h>
#include <gpusim/update_ops_cuda.h>

void test_single_diagonal_matrix_gate(std::function<void(UINT, const CTYPE*, void*, ITYPE)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	Eigen::MatrixXcd Identity(2, 2), Z(2, 2);
	Identity << 1, 0, 0, 1;
	Z << 1, 0, 0, -1;

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

	UINT target;
	double icoef, zcoef, norm;

	auto state = allocate_quantum_state_host(dim);
	initialize_Haar_random_state_host(state, dim);
	Eigen::VectorXcd test_state = copy_cpu_from_gpu(state,dim);

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
		state_equal_gpu(state, test_state, dim, "single diagonal gate");
	}
	release_quantum_state_host(state);
}

TEST(UpdateTest, SingleDiagonalMatrixTest) {
	void(*func)(UINT, const CPPCTYPE*, void*, ITYPE)
		= &single_qubit_diagonal_matrix_gate_host;
	test_single_diagonal_matrix_gate(func);
}

void test_single_phase_gate(std::function<void(UINT, CTYPE, void*, ITYPE)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

	UINT target;
	double angle;

	auto state = allocate_quantum_state_host(dim);
	initialize_Haar_random_state_host(state, dim);
	Eigen::VectorXcd test_state = copy_cpu_from_gpu(state,dim);

	Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		// single qubit phase matrix gate
		target = rand_int(n);
		angle = rand_real();
		U << 1, 0, 0, cos(angle) + 1.i*sin(angle);
		func(target, cos(angle) + 1.i*sin(angle), state, dim);
		test_state = get_expanded_eigen_matrix_with_identity(target, U, n) * test_state;
		state_equal_gpu(state, test_state, dim, "single phase gate");
	}
	release_quantum_state_host(state);
}


TEST(UpdateTest, SinglePhaseGateTest) {
	void(*func)(UINT, CTYPE, void*, ITYPE)
		= &single_qubit_phase_gate_host;
	test_single_phase_gate(func);
}
