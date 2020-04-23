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


TEST(CircuitTest, CircuitOptimizeLight) {
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
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

	/*
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
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }
	*/


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
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

	/*
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
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }
	*/

	/*
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
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }
	*/

	/*
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
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }
	*/

	/*
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
        qco.optimize_light(copy_circuit);
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
        qco.optimize_light(copy_circuit);
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
        qco.optimize_light(copy_circuit);
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
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }
	*/


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
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

	/*
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
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }
	*/

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
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }
}

TEST(CircuitTest, RandomCircuitOptimizeLight) {
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
                UINT r = random.int32() % 2 + 3;
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
		QuantumCircuit* copy_circuit = circuit.copy();
		qco.optimize_light(copy_circuit);
		state.load(&org_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << copy_circuit << std::endl;
        for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        delete copy_circuit;
    }

}

TEST(CircuitTest, RandomCircuitOptimizeLight2) {
	const UINT n = 5;
	const UINT dim = 1ULL << n;
	//const UINT depth = 10;
	const UINT depth = 10;
	Random random;
	double eps = 1e-14;
	UINT max_repeat = 3;

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
		QuantumCircuit* copy_circuit = circuit.copy();
		//for (auto gate : copy_circuit->gate_list) {
		//	std::cout << "(";
		//	for (auto val : gate->get_target_index_list()) std::cout << val << " ";
		//	for (auto val : gate->get_control_index_list()) std::cout << val << " ";
		//	std::cout << ")  ";
		//}
		//std::cout << std::endl;
		qco.optimize_light(copy_circuit);
		//for (auto gate : copy_circuit->gate_list) {
		//	std::cout << "(";
		//	for (auto val : gate->get_target_index_list()) std::cout << val << " ";
		//	for (auto val : gate->get_control_index_list()) std::cout << val << " ";
		//	std::cout << ")  ";
		//}
		state.load(&org_state);
		copy_circuit->update_quantum_state(&state);
		//std::cout << copy_circuit << std::endl;
		//for (UINT i = 0; i < dim; ++i) std::cout << state.data_cpp()[i] << "    " << test_state.data_cpp()[i] << std::endl;
		for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
		delete copy_circuit;
	}

}

TEST(CircuitTest, RandomCircuitOptimizeLight3) {
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
		QuantumCircuit* copy_circuit = circuit.copy();
		qco.optimize_light(copy_circuit);
		state.load(&org_state);
		copy_circuit->update_quantum_state(&state);
		//std::cout << copy_circuit << std::endl;
		for (UINT i = 0; i < dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
		delete copy_circuit;
	}
}
