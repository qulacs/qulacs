#include <gtest/gtest.h>
#include "../util/util.h"

#include <cppsim/state_gpu.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/pauli_operator.hpp>

inline void assert_cpu_eq_gpu(QuantumStateCpu& state_cpu, QuantumStateGpu& state_gpu, ITYPE dim, double eps) {
	auto gpu_state_vector = state_gpu.duplicate_data_cpp();
	for (ITYPE i = 0; i < dim; ++i) {
		ASSERT_NEAR(state_cpu.data_cpp()[i].real(), gpu_state_vector[i].real(), eps);
		ASSERT_NEAR(state_cpu.data_cpp()[i].imag(), gpu_state_vector[i].imag(), eps);
	}
	delete gpu_state_vector;
}


TEST(CompatTest, ApplyRandomOrderUnitary) {
	UINT n = 15;
	UINT max_gate_count = 5;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;

	UINT gate_count = 10;
	UINT max_repeat = 3;
	Random random;
	random.set_seed(0);

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);

		// define states
		QuantumStateCpu state_cpu(n);
		QuantumStateGpu state_gpu(n, idx);
		std::vector<UINT> indices(n);
		for (UINT i = 0; i < n; ++i) indices[i] = i;
		std::mt19937 engine(0);

		for (UINT m = 1; m <= max_gate_count; ++m) {
			if (m <= 5) { gate_count = 10; max_repeat = 3; }
			else { gate_count = 1; max_repeat = 1; }
			for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
				state_gpu.set_Haar_random_state();
				state_cpu.load(&state_gpu);

				for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
					std::shuffle(indices.begin(), indices.end(), engine);
					std::vector<UINT> targets(indices.begin(), indices.begin() + m);
					//for (UINT i : targets) std::cout << i << " "; std::cout << std::endl;
					QuantumGateBase* gate = NULL;

					// for small matrix, use random unitary
					if (m <= 7) gate = gate::RandomUnitary(targets);
					// for large matrix, use pauli rotation
					else {
						std::vector<UINT> paulis(m);
						for (UINT i = 0; i < m; ++i) paulis[i] = random.int32() % 4;
						gate = gate::PauliRotation(targets, paulis, 3.14159 * random.uniform());
						QuantumGateBase* new_gate = gate::to_matrix_gate(gate);
						delete gate;
						gate = new_gate;
					}
					gate->update_quantum_state(&state_cpu);
					gate->update_quantum_state(&state_gpu);
					delete gate;
				}

				assert_cpu_eq_gpu(state_cpu, state_gpu, dim, eps);
			}
		}
	}
}


TEST(CompatTest, SingleControlQubitApplyRandomOrderUnitary) {
	UINT n = 15;
	UINT max_gate_count = 5;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;

	UINT gate_count = 10;
	UINT max_repeat = 3;
	Random random;
	random.set_seed(0);

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);

		// define states
		QuantumStateCpu state_cpu(n);
		QuantumStateGpu state_gpu(n, idx);
		std::vector<UINT> indices(n);
		for (UINT i = 0; i < n; ++i) indices[i] = i;
		std::mt19937 engine(0);

		for (UINT m = 1; m <= max_gate_count; ++m) {
			if (m <= 5) { gate_count = 10; max_repeat = 3; }
			else { gate_count = 1; max_repeat = 1; }
			for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
				state_gpu.set_Haar_random_state();
				state_cpu.load(&state_gpu);

				for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
					std::shuffle(indices.begin(), indices.end(), engine);
					std::vector<UINT> targets(indices.begin(), indices.begin() + m);
					//for (UINT i : targets) std::cout << i << " "; std::cout << std::endl;
					QuantumGateMatrix* gate = NULL;

					gate = gate::RandomUnitary(targets);
					gate->add_control_qubit(indices[m], 1);
					gate->update_quantum_state(&state_cpu);
					gate->update_quantum_state(&state_gpu);
					delete gate;
				}

				assert_cpu_eq_gpu(state_cpu, state_gpu, dim, eps);
			}
		}
	}
}

TEST(CompatTest, DoubleControlQubitApplyRandomOrderUnitary) {
	UINT n = 15;
	UINT max_gate_count = 5;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;

	UINT gate_count = 10;
	UINT max_repeat = 3;
	Random random;
	random.set_seed(0);

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);

		// define states
		QuantumStateCpu state_cpu(n);
		QuantumStateGpu state_gpu(n, idx);
		std::vector<UINT> indices(n);
		for (UINT i = 0; i < n; ++i) indices[i] = i;
		std::mt19937 engine(0);

		for (UINT m = 1; m <= max_gate_count; ++m) {
			if (m <= 5) { gate_count = 10; max_repeat = 3; }
			else { gate_count = 1; max_repeat = 1; }
			for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
				state_gpu.set_Haar_random_state();
				state_cpu.load(&state_gpu);

				for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
					std::shuffle(indices.begin(), indices.end(), engine);
					std::vector<UINT> targets(indices.begin(), indices.begin() + m);
					//for (UINT i : targets) std::cout << i << " "; std::cout << std::endl;
					QuantumGateMatrix* gate = NULL;

					gate = gate::RandomUnitary(targets);
					gate->add_control_qubit(indices[m], 1);
					gate->add_control_qubit(indices[m+1], 1);
					gate->update_quantum_state(&state_cpu);
					gate->update_quantum_state(&state_gpu);
					delete gate;
				}

				assert_cpu_eq_gpu(state_cpu, state_gpu, dim, eps);
			}
		}
	}
}


TEST(CompatTest, MultiControlQubitApplyRandomOrderUnitary) {
	UINT n = 15;
	UINT max_gate_count = 5;
	UINT max_control = 5;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;

	UINT gate_count = 10;
	UINT max_repeat = 3;
	Random random;
	random.set_seed(0);

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);

		// define states
		QuantumStateCpu state_cpu(n);
		QuantumStateGpu state_gpu(n, idx);
		std::vector<UINT> indices(n);
		for (UINT i = 0; i < n; ++i) indices[i] = i;
		std::mt19937 engine(0);

		for (UINT c = 0; c <= max_control; ++c) {
			for (UINT m = 1; m <= max_gate_count; ++m) {
				if (m <= 5) { gate_count = 10; max_repeat = 3; }
				else { gate_count = 1; max_repeat = 1; }
				for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
					state_gpu.set_Haar_random_state();
					state_cpu.load(&state_gpu);

					for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
						std::shuffle(indices.begin(), indices.end(), engine);
						std::vector<UINT> targets(indices.begin(), indices.begin() + m);
						//for (UINT i : targets) std::cout << i << " "; std::cout << std::endl;
						QuantumGateMatrix* gate = NULL;

						gate = gate::RandomUnitary(targets);
						for (int i = m; i <= m + c; ++i) {
							gate->add_control_qubit(indices[i], 1);
						}
						gate->update_quantum_state(&state_cpu);
						gate->update_quantum_state(&state_gpu);
						delete gate;
					}

					assert_cpu_eq_gpu(state_cpu, state_gpu, dim, eps);
				}
			}
		}
	}
}
