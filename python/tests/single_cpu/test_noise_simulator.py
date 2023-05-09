from typing import List, Set, Tuple

import numpy as np

from qulacs import NoiseSimulator, QuantumCircuit, QuantumState
from qulacs.gate import CNOT, CZ, T, sqrtX, sqrtY


class TestNoiseSimulator:
    def test_noise_simulator(self) -> None:
        def get_heavy_output_probability(
            n, depth, error_prob, shots=1000
        ) -> Tuple[float, Set[int], List[int]]:
            one_qubit_noise = [
                "Depolarizing",
                "BitFlip",
                "Dephasing",
                "IndependentXZ",
                "AmplitudeDamping",
            ]
            two_qubit_noise = ["Depolarizing"]

            circuit_with_noise = QuantumCircuit(n)
            circuit_without_noise = QuantumCircuit(n)
            for d in range(depth):
                for i in range(n):
                    r = np.random.randint(0, 5)
                    if r == 0:
                        circuit_with_noise.add_noise_gate(
                            sqrtX(i),
                            one_qubit_noise[np.random.randint(0, 5)],
                            error_prob,
                        )
                        circuit_without_noise.add_sqrtX_gate(i)
                    elif r == 1:
                        circuit_with_noise.add_noise_gate(
                            sqrtY(i),
                            one_qubit_noise[np.random.randint(0, 5)],
                            error_prob,
                        )
                        circuit_without_noise.add_sqrtY_gate(i)
                    elif r == 2:
                        circuit_with_noise.add_noise_gate(
                            T(i), one_qubit_noise[np.random.randint(0, 5)], error_prob
                        )
                        circuit_without_noise.add_T_gate(i)
                    elif r == 3:
                        if i + 1 < n:
                            circuit_with_noise.add_noise_gate(
                                CNOT(i, i + 1),
                                two_qubit_noise[np.random.randint(0, 1)],
                                error_prob,
                            )
                            circuit_without_noise.add_CNOT_gate(i, i + 1)
                    elif r == 4:
                        if i + 1 < n:
                            circuit_with_noise.add_noise_gate(
                                CZ(i, i + 1),
                                two_qubit_noise[np.random.randint(0, 1)],
                                error_prob,
                            )
                            circuit_without_noise.add_CZ_gate(i, i + 1)

            ideal_state = QuantumState(n)
            circuit_without_noise.update_quantum_state(ideal_state)
            prob_dist = [abs(x) ** 2 for x in ideal_state.get_vector()]
            p_median = np.sort(prob_dist)[2 ** (n - 1) - 1]
            heavy_output = set()
            for i in range(2**n):
                if prob_dist[i] > p_median:
                    heavy_output.add(i)

            sim = NoiseSimulator(circuit_with_noise, QuantumState(n))
            noisy_sample = sim.execute(shots)
            num_heavy_output = 0
            for sample in noisy_sample:
                if sample in heavy_output:
                    num_heavy_output += 1
            return num_heavy_output / shots, heavy_output, noisy_sample

        (
            low_noise_prob,
            low_noise_heavy_output,
            low_noise_result,
        ) = get_heavy_output_probability(7, 100, 1e-5)
        (
            high_noise_prob,
            high_noise_heavy_output,
            high_noise_result,
        ) = get_heavy_output_probability(7, 100, 0.01)
        if low_noise_prob < 2 / 3:
            print(
                "[ERROR] On low noise environment Heavy Output percentage should be > 0.666,"
                f"but was {low_noise_prob}"
            )
            print("Telemetry Information:")
            print(f"Sampling Result: {low_noise_result}")
            print(f"Heavy Output: {low_noise_heavy_output}")
        if high_noise_prob > 2 / 3:
            print(
                "[ERROR] On high noise environment Heavy Output percentage should be < 0.666, "
                f"but was {high_noise_prob}"
            )
            print("Telemetry Information:")
            print(f"Sampling Result: {high_noise_result}")
            print(f"Heavy Output: {high_noise_heavy_output}")

        assert low_noise_prob > 2 / 3
        assert high_noise_prob < 2 / 3
