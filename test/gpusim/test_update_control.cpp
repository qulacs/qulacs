
#include "test_util.hpp"
#include <gpusim/memory_ops.h>
#include <gpusim/update_ops_cuda.h>

void test_single_control_single_target(std::function<void(unsigned int, unsigned int, unsigned int, const CPPCTYPE*, void*, ITYPE, void*, UINT)> func) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

    UINT target,control;
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

		for (UINT rep = 0; rep < max_repeat; ++rep) {
			// single qubit control-1 single qubit gate
			target = rand_int(n);
			control = rand_int(n - 1);
			if (control >= target) control++;
			U = get_eigen_matrix_random_single_qubit_unitary();
			func(control, 1, target, (CPPCTYPE*)U.data(), state, dim, stream_ptr, idx);
			test_state = (get_expanded_eigen_matrix_with_identity(control, P0, n) + get_expanded_eigen_matrix_with_identity(control, P1, n) * get_expanded_eigen_matrix_with_identity(target, U, n)) * test_state;
			state_equal_gpu(state, test_state, dim, "single qubit control sinlge qubit dense gate", stream_ptr, idx);

			// single qubit control-0 single qubit gate
			target = rand_int(n);
			control = rand_int(n - 1);
			if (control >= target) control++;
			U = get_eigen_matrix_random_single_qubit_unitary();
			func(control, 0, target, (CPPCTYPE*)U.data(), state, dim, stream_ptr, idx);
			test_state = (get_expanded_eigen_matrix_with_identity(control, P1, n) + get_expanded_eigen_matrix_with_identity(control, P0, n) * get_expanded_eigen_matrix_with_identity(target, U, n)) * test_state;
			state_equal_gpu(state, test_state, dim, "single qubit control sinlge qubit dense gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, SingleQubitControlSingleQubitDenseMatrixTest) {
	void(*func)(UINT, UINT, UINT, const CPPCTYPE*, void*, ITYPE, void*, UINT) 
		= &single_qubit_control_single_qubit_dense_matrix_gate_host;
	test_single_control_single_target(func);
}

/*
void test_two_control_single_target(std::function<void(const UINT*, const UINT*, UINT, UINT, const CPPCTYPE*, void*, ITYPE)> func) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

    UINT target;
    UINT controls[2];

    auto state = allocate_quantum_state_host(dim);
    initialize_Haar_random_state_host(state, dim);
    Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim);

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // two qubit control-10 single qubit gate
        std::random_shuffle(index_list.begin(), index_list.end());
        target = index_list[0];
        controls[0] = index_list[1];
        controls[1] = index_list[2];

        U = get_eigen_matrix_random_single_qubit_unitary();
        UINT mvalues[2] = { 1,0 };
        func(controls, mvalues, 2, target, (CPPCTYPE*)U.data(), state, dim);
        test_state = (
            get_expanded_eigen_matrix_with_identity(controls[0], P0, n)*get_expanded_eigen_matrix_with_identity(controls[1], P0, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P0, n)*get_expanded_eigen_matrix_with_identity(controls[1], P1, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P1, n)*get_expanded_eigen_matrix_with_identity(controls[1], P0, n)*get_expanded_eigen_matrix_with_identity(target, U, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P1, n)*get_expanded_eigen_matrix_with_identity(controls[1], P1, n)
            ) * test_state;
        state_equal(state, test_state, dim, "two qubit control sinlge qubit dense gate");

    }
    release_quantum_state_host(state);
}

TEST(UpdateTest, TwoQubitControlSingleQubitDenseMatrixTest) {
	// not implemented
	//test_two_control_single_target(multi_qubit_control_single_qubit_dense_matrix_gate);
}
*/

TEST(UpdateTest, SingleQubitControlTwoQubitDenseMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U, U2;
    Eigen::Matrix<std::complex<double>, 4, 4, Eigen::RowMajor> Umerge;

    UINT targets[2], control;
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

		for (UINT rep = 0; rep < max_repeat; ++rep) {
			// single qubit 1-controlled qubit dense matrix gate
			U = get_eigen_matrix_random_single_qubit_unitary();
			U2 = get_eigen_matrix_random_single_qubit_unitary();
			std::random_shuffle(index_list.begin(), index_list.end());
			targets[0] = index_list[0];
			targets[1] = index_list[1];
			control = index_list[2];

			Umerge = kronecker_product(U2, U);
			test_state = (get_expanded_eigen_matrix_with_identity(control, P0, n) + get_expanded_eigen_matrix_with_identity(control, P1, n) * get_expanded_eigen_matrix_with_identity(targets[1], U2, n) * get_expanded_eigen_matrix_with_identity(targets[0], U, n)) * test_state;
			single_qubit_control_multi_qubit_dense_matrix_gate_host(control, 1, targets, 2, (CPPCTYPE*)Umerge.data(), state, dim, stream_ptr, idx);
			state_equal_gpu(state, test_state, dim, "single qubit control two-qubit separable dense gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}



TEST(UpdateTest, TwoQubitControlTwoQubitDenseMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U, U2;
    Eigen::Matrix<std::complex<double>, 4, 4, Eigen::RowMajor> Umerge;

    UINT targets[2], controls[2],mvalues[2];
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

		for (UINT rep = 0; rep < max_repeat; ++rep) {

			// two qubit control-11 two qubit gate
			U = get_eigen_matrix_random_single_qubit_unitary();
			U2 = get_eigen_matrix_random_single_qubit_unitary();
			std::random_shuffle(index_list.begin(), index_list.end());
			targets[0] = index_list[0];
			targets[1] = index_list[1];
			controls[0] = index_list[2];
			controls[1] = index_list[3];

			mvalues[0] = 1; mvalues[1] = 1;
			Umerge = kronecker_product(U2, U);
			multi_qubit_control_multi_qubit_dense_matrix_gate_host(controls, mvalues, 2, targets, 2, (CPPCTYPE*)Umerge.data(), state, dim, stream_ptr, idx);
			test_state = (
				get_expanded_eigen_matrix_with_identity(controls[0], P0, n) * get_expanded_eigen_matrix_with_identity(controls[1], P0, n) +
				get_expanded_eigen_matrix_with_identity(controls[0], P0, n) * get_expanded_eigen_matrix_with_identity(controls[1], P1, n) +
				get_expanded_eigen_matrix_with_identity(controls[0], P1, n) * get_expanded_eigen_matrix_with_identity(controls[1], P0, n) +
				get_expanded_eigen_matrix_with_identity(controls[0], P1, n) * get_expanded_eigen_matrix_with_identity(controls[1], P1, n) * get_expanded_eigen_matrix_with_identity(targets[0], U, n) * get_expanded_eigen_matrix_with_identity(targets[1], U2, n)
				) * test_state;
			state_equal_gpu(state, test_state, dim, "two qubit control two qubit dense gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}
