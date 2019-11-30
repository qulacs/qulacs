
#include "test_util.hpp"
#include <gpusim/memory_ops.h>
#include <gpusim/update_ops_cuda.h>

void test_single_diagonal_matrix_gate(std::function<void(UINT, const CPPCTYPE*, void*, ITYPE, void*, UINT)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	Eigen::MatrixXcd Identity(2, 2), Z(2, 2);
	Identity << 1, 0, 0, 1;
	Z << 1, 0, 0, -1;

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

	UINT target;
	double icoef, zcoef, norm;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

		for (UINT rep = 0; rep < max_repeat; ++rep) {
			// single qubit diagonal matrix gate
			target = rand_int(n);
			icoef = rand_real(); zcoef = rand_real();
			norm = sqrt(icoef * icoef + zcoef * zcoef);
			icoef /= norm; zcoef /= norm;
			U = icoef * Identity + 1.i * zcoef * Z;
			Eigen::VectorXcd diag = U.diagonal();
			func(target, (CPPCTYPE*)diag.data(), state, dim, stream_ptr, idx);
			test_state = get_expanded_eigen_matrix_with_identity(target, U, n) * test_state;
			state_equal_gpu(state, test_state, dim, "single diagonal gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, SingleDiagonalMatrixTest) {
	void(*func)(UINT, const CPPCTYPE*, void*, ITYPE, void*, UINT)
		= &single_qubit_diagonal_matrix_gate_host;
	test_single_diagonal_matrix_gate(func);
}

void test_single_phase_gate(std::function<void(UINT, CPPCTYPE, void*, ITYPE, void*, UINT)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

	UINT target;
	double angle;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

		for (UINT rep = 0; rep < max_repeat; ++rep) {
			// single qubit phase matrix gate
			target = rand_int(n);
			angle = rand_real();
			U << 1, 0, 0, cos(angle) + 1.i * sin(angle);
			func(target, cos(angle) + 1.i * sin(angle), state, dim, stream_ptr, idx);
			test_state = get_expanded_eigen_matrix_with_identity(target, U, n) * test_state;
			state_equal_gpu(state, test_state, dim, "single phase gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}


TEST(UpdateTest, SinglePhaseGateTest) {
	void(*func)(UINT, CPPCTYPE, void*, ITYPE, void*, UINT)
		= &single_qubit_phase_gate_host;
	test_single_phase_gate(func);
}
