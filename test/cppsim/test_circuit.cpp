#include <gtest/gtest.h>
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

    QuantumState state(n);
    ComplexVector state_eigen(dim);

    state.set_Haar_random_state();
    for (ITYPE i = 0; i < dim; ++i) state_eigen[i] = state.data_cpp()[i];

    QuantumCircuit circuit(n);
    UINT target,target_sub;
    double angle;

    target = random.int32() % n;
    circuit.add_X_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, get_eigen_matrix_single_Pauli(1),n)*state_eigen;

    target = random.int32() % n;
    circuit.add_Y_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, get_eigen_matrix_single_Pauli(2), n)*state_eigen;

    target = random.int32() % n;
    circuit.add_Z_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, get_eigen_matrix_single_Pauli(3), n)*state_eigen;

    target = random.int32() % n;
    circuit.add_H_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, H, n)*state_eigen;

    target = random.int32() % n;
    circuit.add_S_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, S, n)*state_eigen;

    target = random.int32() % n;
    circuit.add_Sdag_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, S.adjoint(), n)*state_eigen;

    target = random.int32() % n;
    circuit.add_T_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, T, n)*state_eigen;

    target = random.int32() % n;
    circuit.add_Tdag_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, T.adjoint(), n)*state_eigen;

    target = random.int32() % n;
    circuit.add_sqrtX_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, sqrtX, n)*state_eigen;

    target = random.int32() % n;
    circuit.add_sqrtXdag_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, sqrtX.adjoint(), n)*state_eigen;

    target = random.int32() % n;
    circuit.add_sqrtY_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, sqrtY, n)*state_eigen;

    target = random.int32() % n;
    circuit.add_sqrtYdag_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, sqrtY.adjoint(), n)*state_eigen;

    target = random.int32() % n;
    circuit.add_P0_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, P0, n)*state_eigen;

    target = (target+1)%n;
    circuit.add_P1_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, P1, n)*state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RX_gate(target,angle);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, cos(angle/2)*Identity + 1.i*sin(angle/2)*X, n)*state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RY_gate(target, angle);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, cos(angle/2)*Identity + 1.i*sin(angle/2)*Y, n)*state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RZ_gate(target, angle);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, cos(angle/2)*Identity + 1.i*sin(angle/2)*Z, n)*state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n-1);
    if (target_sub >= target) target_sub++;
    circuit.add_CNOT_gate(target, target_sub);
    state_eigen = get_eigen_matrix_full_qubit_CNOT(target,target_sub,n)*state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_CZ_gate(target, target_sub);
    state_eigen = get_eigen_matrix_full_qubit_CZ(target, target_sub, n)*state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_SWAP_gate(target, target_sub);
    state_eigen = get_eigen_matrix_full_qubit_SWAP(target, target_sub, n)*state_eigen;

    circuit.update_quantum_state(&state);
    for (ITYPE i = 0; i < dim; ++i) ASSERT_NEAR(abs(state_eigen[i] - state.data_cpp()[i]), 0, eps);
}


TEST(CircuitTest, CircuitOptimize) {
    const UINT n = 4;
    const UINT dim = 1ULL << n;
    double eps = 1e-14;

    {
        // merge successive gates
        QuantumState state(n), test_state(n);
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
        qco.optimize(copy_circuit,block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // tensor product, merged
        QuantumState state(n), test_state(n);
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
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }


    {
        // do not take tensor product
        QuantumState state(n), test_state(n);
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
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, control does not commute with X
        QuantumState state(n), test_state(n);
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
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, control does not commute with Z
        QuantumState state(n), test_state(n);
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
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, control commute with Z
        QuantumState state(n), test_state(n);
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
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n), test_state(n);
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
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n), test_state(n);
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
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n), test_state(n);
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
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n), test_state(n);
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
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }


    {
        // CNOT, target commute with X
        QuantumState state(n), test_state(n);
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
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n), test_state(n);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_Z_gate(0);
        circuit.add_gate( gate::merge(gate::CNOT(0,1), gate::Y(2)));
        circuit.add_gate( gate::merge(gate::CNOT(1,0), gate::Y(2)));
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
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n), test_state(n);
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
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
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

    for(UINT repeat=0;repeat<max_repeat;++repeat){
        QuantumState state(n), org_state(n), test_state(n);
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
            for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
            delete copy_circuit;
        }
    }

}

TEST(CircuitTest, RandomCircuitOptimize2) {
	const UINT n = 5;
	const UINT dim = 1ULL << n;
	const UINT depth = 10;
	Random random;
	double eps = 1e-14;
	UINT max_repeat = 3;
	UINT max_block_size = n;

	for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
		QuantumState state(n), org_state(n), test_state(n);
		state.set_Haar_random_state();
		org_state.load(&state);
		QuantumCircuit circuit(n);

		for (UINT d = 0; d < depth; ++d) {
			for (UINT i = 0; i < n; ++i) {
				UINT r = random.int32() % 6;
				if (r == 0)    circuit.add_sqrtX_gate(i);
				else if (r == 1) circuit.add_sqrtY_gate(i);
				else if (r == 2) circuit.add_T_gate(i);
				else if (r == 3) {
					UINT r2 = random.int32() % n;
					if (r2 == i) r2 = (r2 + 1) % n;
					if (i + 1 < n) circuit.add_CNOT_gate(i, r2);
				}
				else if (r == 4) {
					UINT r2 = random.int32() % n;
					if (r2 == i) r2 = (r2 + 1) % n;
					if (i + 1 < n) circuit.add_CZ_gate(i, r2);
				}
				else if (r == 5) {
					UINT r2 = random.int32() % n;
					if (r2 == i) r2 = (r2 + 1) % n;
					if (i + 1 < n) circuit.add_SWAP_gate(i, r2);
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
			for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
			delete copy_circuit;
		}
	}

}

TEST(CircuitTest, RandomCircuitOptimize3) {
	const UINT n = 5;
	const UINT dim = 1ULL << n;
	const UINT depth = 10*n;
	Random random;
	double eps = 1e-14;
	UINT max_repeat = 3;
	UINT max_block_size = n;

	std::vector<UINT> qubit_list;
	for (int i = 0; i < n; ++i) qubit_list.push_back(i);

	for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
		QuantumState state(n), org_state(n), test_state(n);
		state.set_Haar_random_state();
		org_state.load(&state);
		QuantumCircuit circuit(n);

		for (UINT d = 0; d < depth; ++d) {
			std::random_shuffle(qubit_list.begin(), qubit_list.end());
			std::vector<UINT> mylist;
			mylist.push_back(qubit_list[0]);
			mylist.push_back(qubit_list[1]);
			circuit.add_random_unitary_gate(mylist);
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
			for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
			delete copy_circuit;
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

    CPPCTYPE res;
    CPPCTYPE test_res;

    Observable diag_observable(n), non_diag_observable(n), observable(n);
    Eigen::MatrixXcd test_observable;

    QuantumState state(n);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);

    QuantumCircuit circuit(n);
    Eigen::MatrixXcd test_circuit;

    for (ITYPE i = 0; i < 6; ++i){
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

    num_repeats = (UINT) std::ceil(angle * (double)n* 100.);
    // circuit.add_diagonal_observable_rotation_gate(diag_observable, angle);
    circuit.add_observable_rotation_gate(observable, angle, num_repeats);

    test_circuit = J * angle * test_observable;
    test_circuit = test_circuit.exp();

    state.set_computational_basis(0);
    test_state(0) = 1.;

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state);

    circuit.update_quantum_state(&state);
    test_state = test_circuit * test_state;

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state);
    ASSERT_NEAR(abs(test_res.real() - res.real())/ test_res.real(), 0, 0.01);


    state.set_Haar_random_state(seed);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state.data_cpp()[i];

    test_state = test_circuit * test_state;
    circuit.update_quantum_state(&state);

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state);
    ASSERT_NEAR(abs(test_res.real() - res.real())/ test_res.real(), 0, 0.01);
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

    CPPCTYPE res;
    CPPCTYPE test_res;

    Observable observable(n);
    Eigen::MatrixXcd test_observable;

    QuantumState state(n);
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

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state);

    // for (ITYPE i = 0; i < dim; ++i) ASSERT_NEAR(abs(test_state[i] - state.data_cpp()[i]), 0, eps);
    ASSERT_NEAR(abs(test_res.real() - res.real())/test_res.real(), 0, 0.01);
    ASSERT_NEAR(res.imag(), 0, eps);
    ASSERT_NEAR(test_res.imag(), 0, eps);


    state.set_Haar_random_state();
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state.data_cpp()[i];

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state);

    test_state = test_circuit * test_state;
    circuit.update_quantum_state(&state);

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state);

    // for (ITYPE i = 0; i < dim; ++i) ASSERT_NEAR(abs(test_state[i] - state.data_cpp()[i]), 0, eps);
    ASSERT_NEAR(abs(test_res.real() - res.real())/test_res.real(), 0, 0.01);
    ASSERT_NEAR(res.imag(), 0, eps);
    ASSERT_NEAR(test_res.imag(), 0, eps);

}


TEST(CircuitTest, SpecialGatesToString) {
	QuantumState state(1);
	QuantumCircuit c(1);
	c.add_gate(gate::DepolarizingNoise(0, 0));
	c.update_quantum_state(&state);
	std::string s = c.to_string();
}

