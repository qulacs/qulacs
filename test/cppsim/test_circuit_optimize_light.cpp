#include <gtest/gtest.h>

#include <cppsim/circuit.hpp>
#include <cppsim/circuit_optimizer.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/type.hpp>
#include <cppsim/utility.hpp>
#include <csim/constant.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include <utility>
#include <vqcsim/parametric_circuit.hpp>

#include "../util/util.hpp"

TEST(CircuitTest, CircuitOptimizeLight) {
    const UINT n = 4;
    const UINT dim = 1ULL << n;

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

        ASSERT_STATE_NEAR(state, test_state, eps);
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
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
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
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    struct TestParameter {
        UINT block_size;
        UINT expected_depth;
        UINT expected_gate_count;
    };

    std::vector<TestParameter> parameter_list = {
        /*TestParameter{2u, 3u, 3u}*/  // TODO: this test was commented out,
                                       // but we might have to re-enable it.
        TestParameter{3u, 2u, 2u}};

    for (TestParameter param : parameter_list) {
        // CNOT, target commute with X

        UINT block_size = param.block_size;
        UINT expected_depth = param.expected_depth;
        UINT expected_gate_count = param.expected_gate_count;

        QuantumState state(n), test_state(n);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        auto gate1 = gate::CNOT(0, 1);
        auto gate2 = gate::CNOT(1, 2);
        auto gate3 = gate::Y(2);

        circuit.add_Z_gate(0);
        circuit.add_gate(gate::merge(gate1, gate3));
        circuit.add_gate(gate::merge(gate2, gate3));
        circuit.add_Z_gate(1);

        delete gate1;
        delete gate2;
        delete gate3;
        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize_light(copy_circuit);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }
}

// Regression test for #634.
// This test checks if the optimizer leaves parametric gates as is.
TEST(CircuitTest, CircuitOptimizeLightParameterUnchanged) {
    const UINT n = 2;
    const UINT dim = 1ULL << n;

    QuantumState state(n), test_state(n);
    state.set_Haar_random_state();
    test_state.load(&state);
    ParametricQuantumCircuit circuit(n);

    circuit.add_X_gate(0);
    circuit.add_parametric_RX_gate(0, 0.1);
    circuit.add_parametric_RY_gate(0, 0.1);
    UINT expected_depth = 3;
    UINT expected_gate_count = 3;

    ParametricQuantumCircuit* copy_circuit = circuit.copy();
    QuantumCircuitOptimizer qco;
    qco.optimize_light(copy_circuit);
    circuit.update_quantum_state(&test_state);
    copy_circuit->update_quantum_state(&state);

    ASSERT_STATE_NEAR(state, test_state, eps);
    ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
    ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
    ASSERT_EQ(copy_circuit->get_parameter_count(), 2);
    delete copy_circuit;
}

TEST(CircuitTest, RandomCircuitOptimizeLight) {
    const UINT n = 5;
    const UINT dim = 1ULL << n;
    const UINT depth = 5;
    Random random;

    UINT max_repeat = 3;
    UINT max_block_size = n;

    for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
        QuantumState state(n), org_state(n), test_state(n);
        state.set_Haar_random_state();
        org_state.load(&state);
        QuantumCircuit circuit(n);

        for (UINT d = 0; d < depth; ++d) {
            for (UINT i = 0; i < n; ++i) {
                UINT r = random.int32() % 2 + 3;
                if (r == 0)
                    circuit.add_sqrtX_gate(i);
                else if (r == 1)
                    circuit.add_sqrtY_gate(i);
                else if (r == 2)
                    circuit.add_T_gate(i);
                else if (r == 3) {
                    if (i + 1 < n) circuit.add_CNOT_gate(i, i + 1);
                } else if (r == 4) {
                    if (i + 1 < n) circuit.add_CZ_gate(i, i + 1);
                }
            }
        }

        test_state.load(&org_state);
        circuit.update_quantum_state(&test_state);
        QuantumCircuitOptimizer qco;
        QuantumCircuit* copy_circuit = circuit.copy();
        qco.optimize_light(copy_circuit);
        state.load(&org_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
        delete copy_circuit;
    }
}

TEST(CircuitTest, RandomCircuitOptimizeLight2) {
    const UINT n = 5;
    const UINT dim = 1ULL << n;
    // const UINT depth = 10;
    const UINT depth = 10;
    Random random;

    UINT max_repeat = 3;

    for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
        QuantumState state(n), org_state(n), test_state(n);
        state.set_Haar_random_state();
        org_state.load(&state);
        QuantumCircuit circuit(n);

        for (UINT d = 0; d < depth; ++d) {
            for (UINT i = 0; i < n; ++i) {
                UINT r = random.int32() % 6;
                if (r == 0)
                    circuit.add_sqrtX_gate(i);
                else if (r == 1)
                    circuit.add_sqrtY_gate(i);
                else if (r == 2)
                    circuit.add_T_gate(i);
                else if (r == 3) {
                    UINT r2 = random.int32() % n;
                    if (r2 == i) r2 = (r2 + 1) % n;
                    if (i + 1 < n) circuit.add_CNOT_gate(i, r2);
                } else if (r == 4) {
                    UINT r2 = random.int32() % n;
                    if (r2 == i) r2 = (r2 + 1) % n;
                    if (i + 1 < n) circuit.add_CZ_gate(i, r2);
                } else if (r == 5) {
                    UINT r2 = random.int32() % n;
                    if (r2 == i) r2 = (r2 + 1) % n;
                    if (i + 1 < n) circuit.add_SWAP_gate(i, r2);
                }
            }
        }

        test_state.load(&org_state);
        circuit.update_quantum_state(&test_state);
        QuantumCircuitOptimizer qco;
        QuantumCircuit* copy_circuit = circuit.copy();
        qco.optimize_light(copy_circuit);
        state.load(&org_state);
        copy_circuit->update_quantum_state(&state);
        ASSERT_STATE_NEAR(state, test_state, eps);
        delete copy_circuit;
    }
}

TEST(CircuitTest, RandomCircuitOptimizeLight3) {
    const UINT n = 5;
    const UINT dim = 1ULL << n;
    const UINT depth = 10 * n;
    Random random;

    UINT max_repeat = 3;
    UINT max_block_size = n;

    std::vector<UINT> qubit_list;
    for (int i = 0; i < n; ++i) qubit_list.push_back(i);

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
        QuantumState state(n), org_state(n), test_state(n);
        state.set_Haar_random_state();
        org_state.load(&state);
        QuantumCircuit circuit(n);

        for (UINT d = 0; d < depth; ++d) {
            std::shuffle(qubit_list.begin(), qubit_list.end(), engine);
            std::vector<UINT> mylist;
            mylist.push_back(qubit_list[0]);
            mylist.push_back(qubit_list[1]);
            circuit.add_random_unitary_gate(mylist);
        }

        test_state.load(&org_state);
        circuit.update_quantum_state(&test_state);
        QuantumCircuitOptimizer qco;
        QuantumCircuit* copy_circuit = circuit.copy();
        qco.optimize_light(copy_circuit);
        state.load(&org_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
        delete copy_circuit;
    }
}

// see https://github.com/qulacs/qulacs/pull/514 for details.
TEST(CircuitTest, FusedSWAPregression1) {
    const UINT n = 4;
    const UINT dim = 1ULL << n;

    QuantumState opt_state(n), ref_state(n);
    opt_state.set_Haar_random_state();
    ref_state.load(&opt_state);
    QuantumCircuit circuit(n);

    circuit.add_H_gate(1);
    circuit.add_FusedSWAP_gate(0, 2, 2);
    circuit.add_H_gate(1);

    QuantumCircuit* opt_circuit = circuit.copy();
    QuantumCircuitOptimizer qco;
    qco.optimize_light(opt_circuit);
    circuit.update_quantum_state(&ref_state);
    opt_circuit->update_quantum_state(&opt_state);

    ASSERT_STATE_NEAR(ref_state, opt_state, eps);
    delete opt_circuit;
}
