#include <gtest/gtest.h>

#include <cppsim_experimental/circuit.hpp>
#include <cppsim_experimental/gate.hpp>
#include <cppsim_experimental/noisesimulator.hpp>
#include <cppsim_experimental/state.hpp>

#include "../util/util.h"

TEST(NoiseSimulatorTest, Random_with_State_Test) {
    // Just Check whether they run without Runtime Errors.
    int n = 10, depth = 10;
    StateVector state(n);
    state.set_Haar_random_state();
    QuantumCircuit circuit(n);
    Random random;
    for (UINT d = 0; d < depth; ++d) {
        for (UINT i = 0; i < n; ++i) {
            UINT r = (UINT)(random.int32()) % 5;
            if (r == 0)
                circuit.add_gate_take(gate::sqrtX(i));
            else if (r == 1)
                circuit.add_gate_take(gate::sqrtY(i));
            else if (r == 2)
                circuit.add_gate_take(gate::T(i));
            else if (r == 3) {
                if (i + 1 < n) circuit.add_gate_take(gate::CNOT(i, i + 1));
            } else if (r == 4) {
                if (i + 1 < n) circuit.add_gate_take(gate::CZ(i, i + 1));
            }
        }
    }
    NoiseSimulator hoge(&circuit, &state);
    std::vector<unsigned int> result = hoge.execute(100);
    return;
}

TEST(NoiseSimulatorTest, Random_without_State_Test) {
    // Just Check whether they run without Runtime Errors.
    int n = 10, depth = 10;
    QuantumCircuit circuit(n);
    Random random;
    for (UINT d = 0; d < depth; ++d) {
        for (UINT i = 0; i < n; ++i) {
            UINT r = (UINT)(random.int32()) % 5;
            if (r == 0)
                circuit.add_gate_take(gate::sqrtX(i));
            else if (r == 1)
                circuit.add_gate_take(gate::sqrtY(i));
            else if (r == 2)
                circuit.add_gate_take(gate::T(i));
            else if (r == 3) {
                if (i + 1 < n) circuit.add_gate_take(gate::CNOT(i, i + 1));
            } else if (r == 4) {
                if (i + 1 < n) circuit.add_gate_take(gate::CZ(i, i + 1));
            }
        }
    }
    NoiseSimulator hoge(&circuit);
    std::vector<unsigned int> result = hoge.execute(100);
    return;
}

TEST(NoiseSimulatorTest, H_gate_twice_test) {
    int n = 4;
    QuantumCircuit circuit(n);
    circuit.add_noise_gate(gate::H(0), "Depolarizing", 0.02);
    circuit.add_noise_gate(gate::H(0), "Depolarizing", 0.02);
    NoiseSimulator hoge(&circuit);
    std::vector<unsigned int> result = hoge.execute(10000);
    int cnts[2] = {};
    for (int i = 0; i < result.size(); ++i) {
        cnts[result[i]]++;
    }
    ASSERT_NE(cnts[0], 0);
    ASSERT_NE(cnts[1], 0);
    ASSERT_GT(cnts[0], cnts[1]);
}