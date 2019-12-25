
#include "test_util.hpp"
#include <gpusim/memory_ops.h>
#include <gpusim/update_ops_cuda.h>

void test_single_qubit_named_gate(UINT n, std::string name, std::function<void(UINT, void*, ITYPE, void*, UINT)> func, Eigen::MatrixXcd mat) {
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 2;
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_with_seed_host(state, dim, 0, stream_ptr, idx);

		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);
		std::vector<UINT> indices;
		for (UINT i = 0; i < n; ++i) indices.push_back(i);

		for (UINT rep = 0; rep < max_repeat; ++rep) {
			for (UINT i = 0; i < n; ++i) {
				UINT target = indices[i];
				func(target, state, dim, stream_ptr, idx);
				test_state = get_expanded_eigen_matrix_with_identity(target, mat, n) * test_state;
				state_equal_gpu(state, test_state, dim, name, stream_ptr, idx);
			}
			std::random_shuffle(indices.begin(), indices.end());
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, XGate) {
	Eigen::MatrixXcd mat(2, 2);
	mat << 0, 1, 1, 0;
	void(*func)(UINT, void*, ITYPE, void*, UINT) = &X_gate_host;
	test_single_qubit_named_gate(6, "XGate", func, mat);
}
TEST(UpdateTest, YGate) {
	Eigen::MatrixXcd mat(2, 2);
	mat << 0, -1.i, 1.i, 0;
	void(*func)(UINT, void*, ITYPE, void*, UINT) = &Y_gate_host;
	test_single_qubit_named_gate(6, "YGate", func, mat);
}
TEST(UpdateTest, ZGate) {
	const UINT n = 3;
	Eigen::MatrixXcd mat(2, 2);
	mat << 1, 0, 0, -1;
	void(*func)(UINT, void*, ITYPE, void*, UINT) = &Z_gate_host;
	test_single_qubit_named_gate(6, "ZGate", func, mat);
}
TEST(UpdateTest, HGate) {
	const UINT n = 3;
	Eigen::MatrixXcd mat(2, 2);
	mat << 1, 1, 1, -1; mat /= sqrt(2.);
	void(*func)(UINT, void*, ITYPE, void*, UINT) = &H_gate_host;
	test_single_qubit_named_gate(n, "HGate", func, mat);
}

TEST(UpdateTest, SGate) {
	const UINT n = 3;
	Eigen::MatrixXcd mat(2, 2);
	mat << 1, 0, 0, 1.i;
	void(*func)(UINT, void*, ITYPE, void*, UINT) = &S_gate_host;
	test_single_qubit_named_gate(n, "SGate", func, mat);
	void(*func2)(UINT, void*, ITYPE, void*, UINT) = &Sdag_gate_host;
	test_single_qubit_named_gate(n, "SGate", func2, mat.adjoint());
}

TEST(UpdateTest, TGate) {
	const UINT n = 3;
	Eigen::MatrixXcd mat(2, 2);
	mat << 1, 0, 0, (1. + 1.i) / sqrt(2.);
	void(*func)(UINT, void*, ITYPE, void*, UINT) = &T_gate_host;
	test_single_qubit_named_gate(n, "TGate", func, mat);
	void(*func2)(UINT, void*, ITYPE, void*, UINT) = &Tdag_gate_host;
	test_single_qubit_named_gate(n, "TGate", func2, mat.adjoint());
}

TEST(UpdateTest, sqrtXGate) {
	const UINT n = 3;
	Eigen::MatrixXcd mat(2, 2);
	mat << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
	void(*func)(UINT, void*, ITYPE, void*, UINT) = &sqrtX_gate_host;
	test_single_qubit_named_gate(n, "SqrtXGate", func, mat);
	void(*func2)(UINT, void*, ITYPE, void*, UINT) = &sqrtXdag_gate_host;
	test_single_qubit_named_gate(n, "SqrtXGate", func2, mat.adjoint());
}

TEST(UpdateTest, sqrtYGate) {
	const UINT n = 3;
	Eigen::MatrixXcd mat(2, 2);
	mat << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
	void(*func)(UINT, void*, ITYPE, void*, UINT) = &sqrtY_gate_host;
	test_single_qubit_named_gate(n, "SqrtYGate", func, mat);
	void(*func2)(UINT, void*, ITYPE, void*, UINT) = &sqrtYdag_gate_host;
	test_single_qubit_named_gate(n, "SqrtYGate", func2, mat.adjoint());
}

void test_projection_gate(std::function<void(UINT, void*, ITYPE, void*, UINT)> func, std::function<double(UINT, void*, ITYPE, void*, UINT)> prob_func, Eigen::MatrixXcd mat) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;
	UINT target;
	double prob;
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);

		for (UINT rep = 0; rep < max_repeat; ++rep) {
			initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
			Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

			target = rand_int(n);
			// Z-projection operators 
			prob = prob_func(target, state, dim, stream_ptr, idx);
			EXPECT_GT(prob, 1e-10);
			func(target, state, dim, stream_ptr, idx);
			ASSERT_NEAR(state_norm_squared_host(state, dim, stream_ptr, idx), prob, eps);
			normalize_host(prob, state, dim, stream_ptr, idx);

			test_state = get_expanded_eigen_matrix_with_identity(target, mat, n) * test_state;
			ASSERT_NEAR(test_state.squaredNorm(), prob, eps);
			test_state.normalize();
			state_equal_gpu(state, test_state, dim, "Projection gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, ProjectionAndNormalizeTest) {
	Eigen::MatrixXcd P0(2, 2), P1(2, 2);
	P0 << 1, 0, 0, 0;
	P1 << 0, 0, 0, 1;
	void(*func1)(UINT, void*, ITYPE, void*, UINT) = &P0_gate_host;
	double(*func2)(UINT, void*, ITYPE, void*, UINT) = &M0_prob_host;
	test_projection_gate(func1,func2,P0);
	func1 = &P1_gate_host;
	func2 = &M1_prob_host;
	test_projection_gate(func1, func2, P1);
}

TEST(UpdateTest, SingleQubitRotationGateTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);
	Identity << 1, 0, 0, 1;
	X << 0, 1, 1, 0;
	Y << 0, -1.i, 1.i, 0;
	Z << 1, 0, 0, -1;

	UINT target;
	double angle;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);
		typedef std::tuple<std::function<void(UINT, double, void*, ITYPE, void*, UINT)>, Eigen::MatrixXcd, std::string> testset;
		std::vector<testset> test_list;
		void(*func)(UINT, double, void*, ITYPE, void*, UINT) = &RX_gate_host;
		test_list.push_back(std::make_tuple(func, X, "Xrot"));
		func = &RY_gate_host;
		test_list.push_back(std::make_tuple(func, Y, "Yrot"));
		func = &RZ_gate_host;
		test_list.push_back(std::make_tuple(func, Z, "Zrot"));

		for (UINT rep = 0; rep < max_repeat; ++rep) {
			for (auto tup : test_list) {
				target = rand_int(n);
				angle = rand_real();
				auto func = std::get<0>(tup);
				auto mat = std::get<1>(tup);
				auto name = std::get<2>(tup);
				func(target, angle, state, dim, stream_ptr, idx);
				test_state = get_expanded_eigen_matrix_with_identity(target, cos(angle / 2) * Identity + 1.i * sin(angle / 2) * mat, n) * test_state;
				state_equal_gpu(state, test_state, dim, name, stream_ptr, idx);
			}
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

void test_two_qubit_named_gate(UINT n, std::string name, std::function<void(UINT, UINT, void*, ITYPE, void*, UINT)> func,
	std::function<Eigen::MatrixXcd(UINT, UINT, UINT)> matfunc) {
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 2;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_with_seed_host(state, dim, 0, stream_ptr, idx);

		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);
		std::vector<UINT> indices;
		for (UINT i = 0; i < n; ++i) indices.push_back(i);

		for (UINT rep = 0; rep < max_repeat; ++rep) {
			for (UINT i = 0; i + 1 < n; i += 2) {
				UINT target = indices[i];
				UINT control = indices[i + 1];
				func(control, target, state, dim, stream_ptr, idx);
				Eigen::MatrixXcd mat = matfunc(control, target, n);
				test_state = mat * test_state;
				state_equal_gpu(state, test_state, dim, name, stream_ptr, idx);
			}
			std::random_shuffle(indices.begin(), indices.end());
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, CNOTGate) {
	const UINT n = 4;
	void(*func)(UINT, UINT, void*, ITYPE, void*, UINT) = &CNOT_gate_host;
	test_two_qubit_named_gate(n, "CNOT", func, get_eigen_matrix_full_qubit_CNOT);
}

TEST(UpdateTest, CZGate) {
	const UINT n = 4;
	void(*func)(UINT, UINT, void*, ITYPE, void*, UINT) = &CZ_gate_host;
	test_two_qubit_named_gate(n, "CZ", func, get_eigen_matrix_full_qubit_CZ);
}

TEST(UpdateTest, SWAPGate) {
	const UINT n = 4;
	void(*func)(UINT, UINT, void*, ITYPE, void*, UINT) = &SWAP_gate_host;
	test_two_qubit_named_gate(n, "SWAP", func, get_eigen_matrix_full_qubit_SWAP);
}
