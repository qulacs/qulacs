#include <gtest/gtest.h>
#include <cppsim/state_gpu.hpp>

#include "../util/util.h"
#include <cppsim/type.hpp>
#include <csim/constant.h>
//#define _USE_MATH_DEFINES
//#include <cmath>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/utility.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/circuit_optimizer.hpp>
#include <utility>
#include <unsupported/Eigen/MatrixFunctions>
#include <cppsim/observable.hpp>
#include <cppsim/pauli_operator.hpp>


inline void set_eigen_from_gpu(ComplexVector& dst, QuantumStateGpu& src, ITYPE dim) {
	auto ptr = src.duplicate_data_cpp();
	for (ITYPE i = 0; i < dim; ++i) dst[i] = ptr[i];
	free(ptr);
}

inline void assert_eigen_eq_gpu(ComplexVector& v1, QuantumStateGpu& v2, ITYPE dim, double eps) {
	auto ptr = v2.duplicate_data_cpp();
	for (UINT i = 0; i < dim; ++i) {
		ASSERT_NEAR(ptr[i].real(), v1[i].real(), eps);
		ASSERT_NEAR(ptr[i].imag(), v1[i].imag(), eps);
	}
	free(ptr);
}

inline void assert_cpu_eq_gpu(QuantumStateCpu& state_cpu, QuantumStateGpu& state_gpu, ITYPE dim, double eps) {
	auto gpu_state_vector = state_gpu.duplicate_data_cpp();
	for (ITYPE i = 0; i < dim; ++i) {
		ASSERT_NEAR(state_cpu.data_cpp()[i].real(), gpu_state_vector[i].real(), eps);
		ASSERT_NEAR(state_cpu.data_cpp()[i].imag(), gpu_state_vector[i].imag(), eps);
	}
	free(gpu_state_vector);
}

inline void assert_gpu_eq_gpu(QuantumStateGpu& state_gpu1, QuantumStateGpu& state_gpu2, ITYPE dim, double eps) {
	auto gpu_state_vector1 = state_gpu1.duplicate_data_cpp();
	auto gpu_state_vector2 = state_gpu2.duplicate_data_cpp();
	for (ITYPE i = 0; i < dim; ++i) {
		ASSERT_NEAR(gpu_state_vector1[i].real(), gpu_state_vector2[i].real(), eps);
		ASSERT_NEAR(gpu_state_vector1[i].imag(), gpu_state_vector2[i].imag(), eps);
	}
	free(gpu_state_vector1);
	free(gpu_state_vector2);
}


TEST(CircuitTest, CircuitBasic) {
    Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2), H(2, 2), S(2, 2), T(2, 2), sqrtX(2, 2), sqrtY(2, 2), P0(2, 2), P1(2, 2);

    Identity << 1, 0, 0, 1;
    X << 0, 1, 1, 0;
    Y << 0, -1.i, 1.i, 0;
    Z << 1, 0, 0, -1;
    H << 1, 1, 1, -1; H /= sqrt(2.);
    S << 1, 0, 0, 1.i;
    T << 1, 0, 0, (1. + 1.i) / sqrt(2.);
    sqrtX << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
    sqrtY << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    const UINT n = 4;
    const UINT dim = 1ULL << n;
    double eps = 1e-14;
    Random random;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);

		QuantumStateGpu state(n, idx);
		ComplexVector state_eigen(dim);

		state.set_Haar_random_state();
		set_eigen_from_gpu(state_eigen, state, dim);

		QuantumCircuit circuit(n);
		UINT target, target_sub;
		double angle;

		target = random.int32() % n;
		circuit.add_X_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, get_eigen_matrix_single_Pauli(1), n) * state_eigen;

		target = random.int32() % n;
		circuit.add_Y_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, get_eigen_matrix_single_Pauli(2), n) * state_eigen;

		target = random.int32() % n;
		circuit.add_Z_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, get_eigen_matrix_single_Pauli(3), n) * state_eigen;

		target = random.int32() % n;
		circuit.add_H_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, H, n) * state_eigen;
		
		target = random.int32() % n;
		circuit.add_S_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, S, n) * state_eigen;
		
		target = random.int32() % n;
		circuit.add_Sdag_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, S.adjoint(), n) * state_eigen;

		target = random.int32() % n;
		circuit.add_T_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, T, n) * state_eigen;
		
		target = random.int32() % n;
		circuit.add_Tdag_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, T.adjoint(), n) * state_eigen;

		target = random.int32() % n;
		circuit.add_sqrtX_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, sqrtX, n) * state_eigen;

		target = random.int32() % n;
		circuit.add_sqrtXdag_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, sqrtX.adjoint(), n) * state_eigen;

		target = random.int32() % n;
		circuit.add_sqrtY_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, sqrtY, n) * state_eigen;

		target = random.int32() % n;
		circuit.add_sqrtYdag_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, sqrtY.adjoint(), n) * state_eigen;

		target = random.int32() % n;
		circuit.add_P0_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, P0, n) * state_eigen;

		target = (target + 1) % n;
		circuit.add_P1_gate(target);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, P1, n) * state_eigen;

		target = random.int32() % n;
		angle = random.uniform() * 3.14159;
		circuit.add_RX_gate(target, angle);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, cos(angle / 2) * Identity + 1.i * sin(angle / 2) * X, n) * state_eigen;

		target = random.int32() % n;
		angle = random.uniform() * 3.14159;
		circuit.add_RY_gate(target, angle);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, cos(angle / 2) * Identity + 1.i * sin(angle / 2) * Y, n) * state_eigen;

		target = random.int32() % n;
		angle = random.uniform() * 3.14159;
		circuit.add_RZ_gate(target, angle);
		state_eigen = get_expanded_eigen_matrix_with_identity(target, cos(angle / 2) * Identity + 1.i * sin(angle / 2) * Z, n) * state_eigen;

		target = random.int32() % n;
		target_sub = random.int32() % (n - 1);
		if (target_sub >= target) target_sub++;
		circuit.add_CNOT_gate(target, target_sub);
		state_eigen = get_eigen_matrix_full_qubit_CNOT(target, target_sub, n) * state_eigen;

		target = random.int32() % n;
		target_sub = random.int32() % (n - 1);
		if (target_sub >= target) target_sub++;
		circuit.add_CZ_gate(target, target_sub);
		state_eigen = get_eigen_matrix_full_qubit_CZ(target, target_sub, n) * state_eigen;

		target = random.int32() % n;
		target_sub = random.int32() % (n - 1);
		if (target_sub >= target) target_sub++;
		circuit.add_SWAP_gate(target, target_sub);
		state_eigen = get_eigen_matrix_full_qubit_SWAP(target, target_sub, n) * state_eigen;
		
		circuit.update_quantum_state(&state);

		assert_eigen_eq_gpu(state_eigen, state, dim, eps);
	}
}


TEST(CircuitTest, CircuitOptimize) {
    const UINT n = 4;
    const UINT dim = 1ULL << n;
    double eps = 1e-14;

    {
        // merge successive gates
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_X_gate(0);
			circuit.add_Y_gate(0);
			UINT block_size = 2;
			UINT expected_depth = 1;
			UINT expected_gate_count = 1;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }

    {
        // tensor product, merged
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_X_gate(0);
			circuit.add_Y_gate(1);
			UINT block_size = 2;
			UINT expected_depth = 1;
			UINT expected_gate_count = 1;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }


    {
        // do not take tensor product
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_X_gate(0);
			circuit.add_Y_gate(1);
			UINT block_size = 1;
			UINT expected_depth = 1;
			UINT expected_gate_count = 2;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }

    {
        // CNOT, control does not commute with X
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_X_gate(0);
			circuit.add_CNOT_gate(0, 1);
			circuit.add_Y_gate(0);
			UINT block_size = 1;
			UINT expected_depth = 3;
			UINT expected_gate_count = 3;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }

    {
        // CNOT, control does not commute with Z
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_X_gate(0);
			circuit.add_CNOT_gate(0, 1);
			circuit.add_Z_gate(0);
			UINT block_size = 1;
			UINT expected_depth = 2;
			UINT expected_gate_count = 2;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }

    {
        // CNOT, control commute with Z
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_Z_gate(0);
			circuit.add_CNOT_gate(0, 1);
			circuit.add_Z_gate(0);
			UINT block_size = 1;
			UINT expected_depth = 2;
			UINT expected_gate_count = 2;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }

    {
        // CNOT, target commute with X
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_X_gate(1);
			circuit.add_CNOT_gate(0, 1);
			circuit.add_X_gate(1);
			UINT block_size = 1;
			UINT expected_depth = 2;
			UINT expected_gate_count = 2;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }

    {
        // CNOT, target commute with X
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_Z_gate(1);
			circuit.add_CNOT_gate(0, 1);
			circuit.add_X_gate(1);
			UINT block_size = 1;
			UINT expected_depth = 2;
			UINT expected_gate_count = 2;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }

    {
        // CNOT, target commute with X
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_X_gate(1);
			circuit.add_CNOT_gate(0, 1);
			circuit.add_Z_gate(1);
			UINT block_size = 1;
			UINT expected_depth = 2;
			UINT expected_gate_count = 2;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			//std::cout << state << std::endl << test_state << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }

    {
        // CNOT, target commute with X
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_Z_gate(1);
			circuit.add_CNOT_gate(0, 1);
			circuit.add_Z_gate(1);
			UINT block_size = 1;
			UINT expected_depth = 3;
			UINT expected_gate_count = 3;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }


    {
        // CNOT, target commute with X
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_Z_gate(1);
			circuit.add_CNOT_gate(0, 1);
			circuit.add_Z_gate(1);
			UINT block_size = 2;
			UINT expected_depth = 1;
			UINT expected_gate_count = 1;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }

    {
        // CNOT, target commute with X
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_Z_gate(0);
			circuit.add_gate(gate::merge(gate::CNOT(0, 1), gate::Y(2)));
			circuit.add_gate(gate::merge(gate::CNOT(1, 0), gate::Y(2)));
			circuit.add_Z_gate(1);
			UINT block_size = 2;
			UINT expected_depth = 3;
			UINT expected_gate_count = 3;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }

    {
        // CNOT, target commute with X
		int ngpus = get_num_device();
		for (int idx = 0; idx < ngpus; ++idx) {
			set_device(idx);

			QuantumStateGpu state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			test_state.load(&state);
			QuantumCircuit circuit(n);

			circuit.add_Z_gate(0);
			circuit.add_gate(gate::merge(gate::CNOT(0, 1), gate::Y(2)));
			circuit.add_gate(gate::merge(gate::CNOT(1, 0), gate::Y(2)));
			circuit.add_Z_gate(1);
			UINT block_size = 3;
			UINT expected_depth = 1;
			UINT expected_gate_count = 1;

			QuantumCircuit* copy_circuit = circuit.copy();
			QuantumCircuitOptimizer qco;
			qco.optimize(copy_circuit, block_size);
			circuit.update_quantum_state(&test_state);
			copy_circuit->update_quantum_state(&state);
			//std::cout << circuit << std::endl << copy_circuit << std::endl;
			assert_gpu_eq_gpu(state, test_state, dim, eps);
			ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
			ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
			delete copy_circuit;
		}
    }
}

TEST(CircuitTest, RandomCircuitOptimize) {
    const UINT n = 5;
    const UINT dim = 1ULL << n;
    const UINT depth = 5;
    Random random;
    double eps = 1e-14;
    UINT max_repeat=3;
    UINT max_block_size = n;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
			QuantumStateGpu state(n, idx), org_state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			org_state.load(&state);
			QuantumCircuit circuit(n);

			for (UINT d = 0; d < depth; ++d) {
				for (UINT i = 0; i < n; ++i) {
					UINT r = random.int32() % 5;
					if (r == 0)    circuit.add_sqrtX_gate(i);
					else if (r == 1) circuit.add_sqrtY_gate(i);
					else if (r == 2) circuit.add_T_gate(i);
					else if (r == 3) {
						if (i + 1 < n) circuit.add_CNOT_gate(i, i + 1);
					}
					else if (r == 4) {
						if (i + 1 < n) circuit.add_CZ_gate(i, i + 1);
					}
				}
			}

			test_state.load(&org_state);
			circuit.update_quantum_state(&test_state);
			//std::cout << circuit << std::endl;
			QuantumCircuitOptimizer qco;
			for (UINT block_size = 1; block_size <= max_block_size; ++block_size) {
				QuantumCircuit* copy_circuit = circuit.copy();
				qco.optimize(copy_circuit, block_size);
				state.load(&org_state);
				copy_circuit->update_quantum_state(&state);
				// std::cout << copy_circuit << std::endl;
				assert_gpu_eq_gpu(state, test_state, dim, eps);
				delete copy_circuit;
			}
		}
	}
}

TEST(CircuitTest, RandomCircuitOptimizeLarge) {
	const UINT n = 5;
	const UINT dim = 1ULL << n;
	const UINT depth = 5;
	Random random;
	double eps = 1e-14;
	UINT max_repeat = 3;
	UINT max_block_size = n;
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
			QuantumStateGpu state(n, idx), org_state(n, idx), test_state(n, idx);
			state.set_Haar_random_state();
			org_state.load(&state);
			QuantumCircuit circuit(n);

			for (UINT d = 0; d < depth; ++d) {
				for (UINT i = 0; i < n; ++i) {
					UINT r = random.int32() % 5;
					if (r == 0)    circuit.add_sqrtX_gate(i);
					else if (r == 1) circuit.add_sqrtY_gate(i);
					else if (r == 2) circuit.add_T_gate(i);
					else if (r == 3) {
						if (i + 1 < n) circuit.add_CNOT_gate(i, i + 1);
					}
					else if (r == 4) {
						if (i + 1 < n) circuit.add_CZ_gate(i, i + 1);
					}
				}
			}

			test_state.load(&org_state);
			circuit.update_quantum_state(&test_state);
			//std::cout << circuit << std::endl;
			QuantumCircuitOptimizer qco;
			for (UINT block_size = 1; block_size <= max_block_size; ++block_size) {
				QuantumCircuit* copy_circuit = circuit.copy();
				qco.optimize(copy_circuit, block_size);
				state.load(&org_state);
				copy_circuit->update_quantum_state(&state);
				//std::cout << copy_circuit << std::endl;
				assert_gpu_eq_gpu(state, test_state, dim, eps);
				delete copy_circuit;
			}
		}
	}
}

TEST(CircuitTest, SuzukiTrotterExpansion) {
    CPPCTYPE J(0.0, 1.0);
    Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);
    Identity << 1, 0, 0, 1;
    X << 0, 1, 1, 0;
    Y << 0, -J, J, 0;
    Z << 1, 0, 0, -1;

    const UINT n = 2;
    UINT num_repeats;
    const UINT dim = 1ULL << n;
    const double eps = 1e-14;

    double angle;
    std::vector<double> coef;

    const UINT seed = 1918;
    Random random;
    random.set_seed(seed);

    double res;
    CPPCTYPE test_res;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		Observable diag_observable(n), non_diag_observable(n), observable(n);
		Eigen::MatrixXcd test_observable;

		QuantumStateGpu state(n, idx);
		Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);

		QuantumCircuit circuit(n);
		Eigen::MatrixXcd test_circuit;

		for (ITYPE i = 0; i < 6; ++i) {
			coef.push_back(-random.uniform());
			// coef.push_back(-1.);
		}
		angle = 2 * PI * random.uniform();


		observable.add_operator(coef[0], "Z 0 I 1");
		observable.add_operator(coef[1], "X 0 Y 1");
		observable.add_operator(coef[2], "Z 0 Z 1");
		observable.add_operator(coef[3], "Z 0 X 1");
		observable.add_operator(coef[4], "Y 0 X 1");
		observable.add_operator(coef[5], "I 0 Z 1");

		test_observable = coef[0] * get_expanded_eigen_matrix_with_identity(0, Z, n);
		test_observable += coef[1] * kronecker_product(Y, X);
		test_observable += coef[2] * kronecker_product(Z, Z);
		test_observable += coef[3] * kronecker_product(X, Z);
		test_observable += coef[4] * kronecker_product(X, Y);
		test_observable += coef[5] * get_expanded_eigen_matrix_with_identity(1, Z, n);

		num_repeats = (UINT)std::ceil(angle * (double)n * 100.);
		// circuit.add_diagonal_observable_rotation_gate(diag_observable, angle);
		circuit.add_observable_rotation_gate(observable, angle, num_repeats);

		test_circuit = J * angle * test_observable;
		test_circuit = test_circuit.exp();

		state.set_computational_basis(0);
		test_state(0) = 1.;

		res = observable.get_expectation_value(&state).real();
		test_res = (test_state.adjoint() * test_observable * test_state);

		circuit.update_quantum_state(&state);
		test_state = test_circuit * test_state;

		res = observable.get_expectation_value(&state).real();
		test_res = (test_state.adjoint() * test_observable * test_state);
		ASSERT_NEAR(abs(test_res.real() - res) / res, 0, 0.01);


		state.set_Haar_random_state(seed);
		set_eigen_from_gpu(test_state, state, dim);

		test_state = test_circuit * test_state;
		circuit.update_quantum_state(&state);

		res = observable.get_expectation_value(&state).real();
		test_res = (test_state.adjoint() * test_observable * test_state);
		ASSERT_NEAR(abs(test_res.real() - res) / res, 0, 0.01);
	}
}

TEST(CircuitTest, RotateDiagonalObservable){
    CPPCTYPE J(0.0, 1.0);
    Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);
    Identity << 1, 0, 0, 1;
    X << 0, 1, 1, 0;
    Y << 0, -J, J, 0;
    Z << 1, 0, 0, -1;

    const UINT n = 2;
    const UINT dim = 1ULL << n;
    const double eps = 1e-14;

    double angle, coef1, coef2;
    Random random;

    double res;
    CPPCTYPE test_res;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		Observable observable(n);
		Eigen::MatrixXcd test_observable;

		QuantumStateGpu state(n, idx);
		Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);

		QuantumCircuit circuit(n);
		Eigen::MatrixXcd test_circuit;

		coef1 = -random.uniform();
		coef2 = -random.uniform();
		angle = 2 * PI * random.uniform();


		observable.add_operator(coef1, "Z 0");
		observable.add_operator(coef2, "Z 0 Z 1");

		test_observable = coef1 * get_expanded_eigen_matrix_with_identity(0, Z, n);
		test_observable += coef2 * kronecker_product(Z, Z);

		circuit.add_diagonal_observable_rotation_gate(observable, angle);
		test_circuit = (J * angle * test_observable).exp();

		state.set_computational_basis(0);
		test_state(0) = 1.;

		// for (ITYPE i = 0; i < dim; ++i) ASSERT_NEAR(abs(test_state[i] - state.data_cpp()[i]), 0, eps);

		circuit.update_quantum_state(&state);
		test_state = test_circuit * test_state;

		res = observable.get_expectation_value(&state).real();
		test_res = (test_state.adjoint() * test_observable * test_state);

		// for (ITYPE i = 0; i < dim; ++i) ASSERT_NEAR(abs(test_state[i] - state.data_cpp()[i]), 0, eps);
		ASSERT_NEAR(abs(test_res.real() - res) / test_res.real(), 0, 0.01);
		ASSERT_NEAR(test_res.imag(), 0, eps);

		state.set_Haar_random_state();
		set_eigen_from_gpu(test_state, state, dim);

		res = observable.get_expectation_value(&state).real();
		test_res = (test_state.adjoint() * test_observable * test_state);

		test_state = test_circuit * test_state;
		circuit.update_quantum_state(&state);

		res = observable.get_expectation_value(&state).real();
		test_res = (test_state.adjoint() * test_observable * test_state);

		// for (ITYPE i = 0; i < dim; ++i) ASSERT_NEAR(abs(test_state[i] - state.data_cpp()[i]), 0, eps);
		ASSERT_NEAR(abs(test_res.real() - res) / test_res.real(), 0, 0.01);
		ASSERT_NEAR(test_res.imag(), 0, eps);
	}
}


TEST(CircuitTest, SpecialGatesToString) {
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		QuantumStateGpu state(1, idx);
		QuantumCircuit c(1);
		c.add_gate(gate::DepolarizingNoise(0, 0));
		c.update_quantum_state(&state);
		std::string s = c.to_string();
	}
}

