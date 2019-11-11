
#include "test_util.hpp"
#include <gpusim/memory_ops.h>
#include <gpusim/update_ops_cuda.h>

void test_single_dense_matrix_gate(std::function<void(UINT, const CPPCTYPE*, void*, ITYPE, void*, UINT)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

	UINT target;
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

		for (UINT rep = 0; rep < max_repeat; ++rep) {
			// single qubit dense matrix gate
			// NOTE: Eigen uses column major by default. To use raw-data of eigen matrix, we need to specify RowMajor.
			target = rand_int(n);
			U = get_eigen_matrix_random_single_qubit_unitary();
			func(target, (CPPCTYPE*)U.data(), state, dim, stream_ptr, idx);
			test_state = get_expanded_eigen_matrix_with_identity(target, U, n) * test_state;
			state_equal_gpu(state, test_state, dim, "single dense gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, SingleDenseMatrixTest) {
	void(*func)(UINT, const CPPCTYPE*, void*, ITYPE, void*, UINT)
		= &single_qubit_dense_matrix_gate_host;
	test_single_dense_matrix_gate(func);
}


void test_double_dense_matrix_gate(std::function<void(UINT, UINT, const CPPCTYPE*, void*, ITYPE, void*, UINT)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	std::vector<UINT> index_list;
	for (UINT i = 0; i < n; ++i) index_list.push_back(i);

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U, U2;
	Eigen::Matrix<std::complex<double>, 4, 4, Eigen::RowMajor> Umerge;

	UINT targets[2];
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

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
			func(targets[0], targets[1], (CPPCTYPE*)Umerge.data(), state, dim, stream_ptr, idx);
			state_equal_gpu(state, test_state, dim, "two-qubit separable dense gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, TwoQubitDenseMatrixTest) {
	void(*func)(UINT,UINT, const CPPCTYPE*, void*, ITYPE, void*, UINT)
		= &double_qubit_dense_matrix_gate_host;
	test_double_dense_matrix_gate(func);
}

void test_three_dense_matrix_gate(std::function<void(UINT, UINT, UINT, const CPPCTYPE*, void*, ITYPE, void*, UINT)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	std::vector<UINT> index_list;
	for (UINT i = 0; i < n; ++i) index_list.push_back(i);

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U1, U2, U3;
	UINT targets[3];
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);

		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

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
			func(targets[0], targets[1], targets[2], (CPPCTYPE*)Umerge.data(), state, dim, stream_ptr, idx);
			state_equal_gpu(state, test_state, dim, "three-qubit separable dense gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, ThreeQubitDenseMatrixTest) {
	void(*func)(UINT, UINT,UINT, const CPPCTYPE*, void*, ITYPE, void*, UINT)
		= &triple_qubit_dense_matrix_gate_host;
	test_three_dense_matrix_gate(func);
}




void test_quad_dense_matrix_gate(std::function<void(const UINT*,  const CPPCTYPE*, void*, ITYPE, void*, UINT)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	std::vector<UINT> index_list;
	for (UINT i = 0; i < n; ++i) index_list.push_back(i);

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U1, U2, U3,U4;
	UINT targets[4];
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);

		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

		Eigen::Matrix<std::complex<double>, 16, 16, Eigen::RowMajor> Umerge;

		for (UINT rep = 0; rep < max_repeat; ++rep) {

			// two qubit dense matrix gate
			U1 = get_eigen_matrix_random_single_qubit_unitary();
			U2 = get_eigen_matrix_random_single_qubit_unitary();
			U3 = get_eigen_matrix_random_single_qubit_unitary();
			U4 = get_eigen_matrix_random_single_qubit_unitary();

			std::random_shuffle(index_list.begin(), index_list.end());
			targets[0] = index_list[0];
			targets[1] = index_list[1];
			targets[2] = index_list[2];
			targets[3] = index_list[3];
			Umerge = kronecker_product(U4, kronecker_product(U3, kronecker_product(U2, U1)));

			test_state =
				get_expanded_eigen_matrix_with_identity(targets[3], U4, n)
				* get_expanded_eigen_matrix_with_identity(targets[2], U3, n)
				* get_expanded_eigen_matrix_with_identity(targets[1], U2, n)
				* get_expanded_eigen_matrix_with_identity(targets[0], U1, n)
				* test_state;
			func(targets, (CPPCTYPE*)Umerge.data(), state, dim, stream_ptr, idx);
			state_equal_gpu(state, test_state, dim, "four-qubit separable dense gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, FourQubitDenseMatrixTest) {
	void(*func)(const UINT*, const CPPCTYPE*, void*, ITYPE, void*, UINT)
		= &quad_qubit_dense_matrix_gate_host;
	test_quad_dense_matrix_gate(func);
}



void test_penta_dense_matrix_gate(std::function<void(const UINT*, const CPPCTYPE*, void*, ITYPE, void*, UINT)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	std::vector<UINT> index_list;
	for (UINT i = 0; i < n; ++i) index_list.push_back(i);

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U1, U2, U3, U4, U5;
	UINT targets[5];
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

		Eigen::Matrix<std::complex<double>, 32, 32, Eigen::RowMajor> Umerge;

		for (UINT rep = 0; rep < max_repeat; ++rep) {

			// two qubit dense matrix gate
			U1 = get_eigen_matrix_random_single_qubit_unitary();
			U2 = get_eigen_matrix_random_single_qubit_unitary();
			U3 = get_eigen_matrix_random_single_qubit_unitary();
			U4 = get_eigen_matrix_random_single_qubit_unitary();
			U5 = get_eigen_matrix_random_single_qubit_unitary();

			std::random_shuffle(index_list.begin(), index_list.end());
			targets[0] = index_list[0];
			targets[1] = index_list[1];
			targets[2] = index_list[2];
			targets[3] = index_list[3];
			targets[4] = index_list[4];
			Umerge = kronecker_product(U5, kronecker_product(U4, kronecker_product(U3, kronecker_product(U2, U1))));

			test_state =
				get_expanded_eigen_matrix_with_identity(targets[4], U5, n)
				* get_expanded_eigen_matrix_with_identity(targets[3], U4, n)
				* get_expanded_eigen_matrix_with_identity(targets[2], U3, n)
				* get_expanded_eigen_matrix_with_identity(targets[1], U2, n)
				* get_expanded_eigen_matrix_with_identity(targets[0], U1, n)
				* test_state;
			func(targets, (CPPCTYPE*)Umerge.data(), state, dim, stream_ptr, idx);
			state_equal_gpu(state, test_state, dim, "five-qubit separable dense gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, FiveQubitDenseMatrixTest) {
	void(*func)(const UINT*, const CPPCTYPE*, void*, ITYPE, void*, UINT)
		= &penta_qubit_dense_matrix_gate_host;
	test_penta_dense_matrix_gate(func);
}



void test_general_dense_matrix_gate(std::function<void(const UINT*, UINT, const CPPCTYPE*, void*, ITYPE, void*, UINT)> func) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	std::vector<UINT> index_list;
	for (UINT i = 0; i < n; ++i) index_list.push_back(i);

	Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U1, U2, U3, U4, U5, U6;
	UINT targets[6];
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);

		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

		Eigen::Matrix<std::complex<double>, 64, 64, Eigen::RowMajor> Umerge;

		for (UINT rep = 0; rep < max_repeat; ++rep) {

			// two qubit dense matrix gate
			U1 = get_eigen_matrix_random_single_qubit_unitary();
			U2 = get_eigen_matrix_random_single_qubit_unitary();
			U3 = get_eigen_matrix_random_single_qubit_unitary();
			U4 = get_eigen_matrix_random_single_qubit_unitary();
			U5 = get_eigen_matrix_random_single_qubit_unitary();
			U6 = get_eigen_matrix_random_single_qubit_unitary();

			std::random_shuffle(index_list.begin(), index_list.end());
			targets[0] = index_list[0];
			targets[1] = index_list[1];
			targets[2] = index_list[2];
			targets[3] = index_list[3];
			targets[4] = index_list[4];
			targets[5] = index_list[5];
			Umerge = kronecker_product(U6, kronecker_product(U5, kronecker_product(U4, kronecker_product(U3, kronecker_product(U2, U1)))));

			test_state =
				get_expanded_eigen_matrix_with_identity(targets[5], U6, n)
				* get_expanded_eigen_matrix_with_identity(targets[4], U5, n)
				* get_expanded_eigen_matrix_with_identity(targets[3], U4, n)
				* get_expanded_eigen_matrix_with_identity(targets[2], U3, n)
				* get_expanded_eigen_matrix_with_identity(targets[1], U2, n)
				* get_expanded_eigen_matrix_with_identity(targets[0], U1, n)
				* test_state;
			func(targets, 6, (CPPCTYPE*)Umerge.data(), state, dim, stream_ptr, idx);
			state_equal_gpu(state, test_state, dim, "six-qubit separable dense gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, SixQubitDenseMatrixTest) {
	void(*func)(const UINT*, UINT, const CPPCTYPE*, void*, ITYPE, void*, UINT)
		= &multi_qubit_dense_matrix_gate_host;
	test_general_dense_matrix_gate(func);
}
