#include <gtest/gtest.h>

#include <cppsim/circuit.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/noisesimulator.hpp>
#include <cppsim/state.hpp>

#include "../util/util.hpp"

TEST(NoiseSimulatorTest, Random_with_State_Test) {
    // Just Check whether they run without Runtime Errors.
    UINT n = 10, depth = 10;
    std::string one_qubit_noise[] = {"Depolarizing", "BitFlip", "Dephasing",
        "IndependentXZ", "AmplitudeDamping"};
    std::string two_qubit_noise[] = {"Depolarizing"};
    QuantumState state(n);
    state.set_Haar_random_state();
    QuantumCircuit circuit_with_noise(n);
    QuantumCircuit circuit_without_noise(n);
    Random random;
    for (UINT d = 0; d < depth; ++d) {
        for (UINT i = 0; i < n; ++i) {
            UINT r = (UINT)(random.int32()) % 5;
            if (r == 0) {
                circuit_with_noise.add_noise_gate(gate::sqrtX(i),
                    one_qubit_noise[(UINT)(random.int32()) % 5], 0.01);
                circuit_without_noise.add_sqrtX_gate(i);
            } else if (r == 1) {
                circuit_with_noise.add_noise_gate(gate::sqrtY(i),
                    one_qubit_noise[(UINT)(random.int32()) % 5], 0.01);
                circuit_without_noise.add_sqrtY_gate(i);
            } else if (r == 2) {
                circuit_with_noise.add_noise_gate(gate::T(i),
                    one_qubit_noise[(UINT)(random.int32()) % 5], 0.01);
                circuit_without_noise.add_T_gate(i);
            } else if (r == 3) {
                if (i + 1 < n) {
                    circuit_with_noise.add_noise_gate(gate::CNOT(i, i + 1),
                        two_qubit_noise[(UINT)(random.int32()) % 1], 0.01);
                    circuit_without_noise.add_CNOT_gate(i, i + 1);
                }
            } else if (r == 4) {
                if (i + 1 < n) {
                    circuit_with_noise.add_noise_gate(gate::CZ(i, i + 1),
                        two_qubit_noise[(UINT)(random.int32()) % 1], 0.01);
                    circuit_without_noise.add_CZ_gate(i, i + 1);
                }
            }
        }
    }

    NoiseSimulator simulator_with_noise(&circuit_with_noise, &state);
    NoiseSimulator simulator_without_noise(&circuit_without_noise, &state);
    std::vector<ITYPE> result_with_noise = simulator_with_noise.execute(100);
    std::vector<ITYPE> result_without_noise =
        simulator_without_noise.execute(100);

    // TODO: compare result_with_noise and result_without_noise and certificate
    // effect of noise
    return;
}

TEST(NoiseSimulatorTest, Random_without_State_Test) {
    // Just Check whether they run without Runtime Errors.
    UINT n = 10, depth = 10;
    std::string one_qubit_noise[] = {"Depolarizing", "BitFlip", "Dephasing",
        "IndependentXZ", "AmplitudeDamping"};
    std::string two_qubit_noise[] = {"Depolarizing"};
    QuantumCircuit circuit_with_noise(n);
    QuantumCircuit circuit_without_noise(n);
    Random random;
    for (UINT d = 0; d < depth; ++d) {
        for (UINT i = 0; i < n; ++i) {
            UINT r = (UINT)(random.int32()) % 5;
            if (r == 0) {
                circuit_with_noise.add_noise_gate(gate::sqrtX(i),
                    one_qubit_noise[(UINT)(random.int32()) % 5], 0.01);
                circuit_without_noise.add_sqrtX_gate(i);
            } else if (r == 1) {
                circuit_with_noise.add_noise_gate(gate::sqrtY(i),
                    one_qubit_noise[(UINT)(random.int32()) % 5], 0.01);
                circuit_without_noise.add_sqrtY_gate(i);
            } else if (r == 2) {
                circuit_with_noise.add_noise_gate(gate::T(i),
                    one_qubit_noise[(UINT)(random.int32()) % 5], 0.01);
                circuit_without_noise.add_T_gate(i);
            } else if (r == 3) {
                if (i + 1 < n) {
                    circuit_with_noise.add_noise_gate(gate::CNOT(i, i + 1),
                        two_qubit_noise[(UINT)(random.int32()) % 1], 0.01);
                    circuit_without_noise.add_CNOT_gate(i, i + 1);
                }
            } else if (r == 4) {
                if (i + 1 < n) {
                    circuit_with_noise.add_noise_gate(gate::CZ(i, i + 1),
                        two_qubit_noise[(UINT)(random.int32()) % 1], 0.01);
                    circuit_without_noise.add_CZ_gate(i, i + 1);
                }
            }
        }
    }

    NoiseSimulator simulator_with_noise(&circuit_with_noise);
    NoiseSimulator simulator_without_noise(&circuit_without_noise);
    std::vector<ITYPE> result_with_noise = simulator_with_noise.execute(100);
    std::vector<ITYPE> result_without_noise =
        simulator_without_noise.execute(100);

    // TODO: compare result_with_noise and result_without_noise and certificate
    // effect of noise
    return;
}

TEST(NoiseSimulatorTest, H_gate_twice_test) {
    UINT n = 4;
    QuantumCircuit circuit(n);
    circuit.add_noise_gate(gate::H(0), "Depolarizing", 0.02);
    circuit.add_noise_gate(gate::H(0), "Depolarizing", 0.02);
    NoiseSimulator sim(&circuit);
    std::vector<ITYPE> result = sim.execute(10000);
    int cnts[2] = {};
    for (UINT i = 0; i < result.size(); ++i) {
        cnts[result[i]]++;
    }
    ASSERT_NE(cnts[0], 0);
    ASSERT_NE(cnts[1], 0);
    ASSERT_GT(cnts[0], cnts[1]);
}
