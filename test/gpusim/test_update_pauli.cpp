
#include "test_util.hpp"
#include <gpusim/memory_ops.h>
#include <gpusim/update_ops_cuda.h>

TEST(UpdateTest, SingleQubitPauliTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	UINT target, pauli;

	Eigen::MatrixXcd Identity(2, 2);
	Identity << 1, 0, 0, 1;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		for (UINT rep = 0; rep < max_repeat; ++rep) {
			target = rand_int(n);
			pauli = rand_int(4);
			single_qubit_Pauli_gate_host(target, pauli, state, dim, stream_ptr, idx);
			test_state = get_expanded_eigen_matrix_with_identity(target, get_eigen_matrix_single_Pauli(pauli), n) * test_state;
			state_equal_gpu(state, test_state, dim, "single Pauli gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, SingleQubitPauliRotationTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	UINT target, pauli;
	double angle;

	Eigen::MatrixXcd Identity(2, 2);
	Identity << 1, 0, 0, 1;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		for (UINT rep = 0; rep < max_repeat; ++rep) {
			target = rand_int(n);
			pauli = rand_int(3) + 1;
			angle = rand_real();
			single_qubit_Pauli_rotation_gate_host(target, pauli, angle, state, dim, stream_ptr, idx);
			test_state = get_expanded_eigen_matrix_with_identity(target, cos(angle / 2) * Identity + 1.i * sin(angle / 2) * get_eigen_matrix_single_Pauli(pauli), n) * test_state;
			state_equal_gpu(state, test_state, dim, "single rotation Pauli gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, MultiQubitPauliTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	UINT pauli;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);
		auto state = allocate_quantum_state_host(dim, idx);
		initialize_Haar_random_state_host(state, dim, stream_ptr, idx);
		Eigen::VectorXcd test_state = copy_cpu_from_gpu(state, dim, stream_ptr, idx);

		for (UINT rep = 0; rep < max_repeat; ++rep) {
			// multi pauli whole
			std::vector<UINT> pauli_whole, pauli_partial, pauli_partial_index;
			std::vector<std::pair<UINT, UINT>> pauli_partial_pair;

			pauli_whole.resize(n);
			for (UINT i = 0; i < n; ++i) {
				pauli_whole[i] = rand_int(4);
			}
			multi_qubit_Pauli_gate_whole_list_host(pauli_whole.data(), n, state, dim, stream_ptr, idx);
			test_state = get_eigen_matrix_full_qubit_pauli(pauli_whole) * test_state;
			state_equal_gpu(state, test_state, dim, "multi Pauli whole gate", stream_ptr, idx);

			// multi pauli partial
			pauli_partial.clear();
			pauli_partial_index.clear();
			pauli_partial_pair.clear();
			for (UINT i = 0; i < n; ++i) {
				pauli = rand_int(4);
				pauli_whole[i] = pauli;
				if (pauli != 0) {
					pauli_partial_pair.push_back(std::make_pair(i, pauli));
				}
			}
			std::random_shuffle(pauli_partial_pair.begin(), pauli_partial_pair.end());
			for (auto val : pauli_partial_pair) {
				pauli_partial_index.push_back(val.first);
				pauli_partial.push_back(val.second);
			}
			multi_qubit_Pauli_gate_partial_list_host(pauli_partial_index.data(), pauli_partial.data(), (UINT)pauli_partial.size(), state, dim, stream_ptr, idx);
			test_state = get_eigen_matrix_full_qubit_pauli(pauli_whole) * test_state;
			state_equal_gpu(state, test_state, dim, "multi Pauli partial gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}

TEST(UpdateTest, MultiQubitPauliRotationTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;

	UINT pauli;
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

			std::vector<UINT> pauli_whole, pauli_partial, pauli_partial_index;
			std::vector<std::pair<UINT, UINT>> pauli_partial_pair;

			// multi pauli rotation whole
			pauli_whole.resize(n);
			for (UINT i = 0; i < n; ++i) {
				pauli_whole[i] = rand_int(4);
			}
			angle = rand_real();
			multi_qubit_Pauli_rotation_gate_whole_list_host(pauli_whole.data(), n, angle, state, dim, stream_ptr, idx);
			test_state = (cos(angle / 2) * whole_I + 1.i * sin(angle / 2) * get_eigen_matrix_full_qubit_pauli(pauli_whole)) * test_state;
			state_equal_gpu(state, test_state, dim, "multi Pauli rotation whole gate", stream_ptr, idx);

			// multi pauli rotation partial
			pauli_partial.clear();
			pauli_partial_index.clear();
			pauli_partial_pair.clear();
			for (UINT i = 0; i < n; ++i) {
				pauli = rand_int(4);
				pauli_whole[i] = pauli;
				if (pauli != 0) {
					pauli_partial_pair.push_back(std::make_pair(i, pauli));
				}
			}
			std::random_shuffle(pauli_partial_pair.begin(), pauli_partial_pair.end());
			for (auto val : pauli_partial_pair) {
				pauli_partial_index.push_back(val.first);
				pauli_partial.push_back(val.second);
			}
			angle = rand_real();
			multi_qubit_Pauli_rotation_gate_partial_list_host(pauli_partial_index.data(), pauli_partial.data(), (UINT)pauli_partial.size(), angle, state, dim, stream_ptr, idx);
			test_state = (cos(angle / 2) * whole_I + 1.i * sin(angle / 2) * get_eigen_matrix_full_qubit_pauli(pauli_whole)) * test_state;
			state_equal_gpu(state, test_state, dim, "multi Pauli rotation partial gate", stream_ptr, idx);
		}
		release_quantum_state_host(state, idx);
		release_cuda_stream_host(stream_ptr, 1, idx);
	}
}
