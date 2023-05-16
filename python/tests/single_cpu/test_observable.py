import numpy as np

from qulacs import GeneralQuantumOperator, Observable


class TestObservable:
    def test_get_matrix_from_observable(self) -> None:
        n_qubits = 3
        obs = Observable(n_qubits)
        obs.add_operator(0.5, "Z 2")
        obs.add_operator(1.0, "X 0 X 1 X 2")
        obs.add_operator(1.0, "Y 1")
        ans = np.array(
            [
                [0.5, 0, -1j, 0, 0, 0, 0, 1],
                [0, 0.5, 0, -1j, 0, 0, 1, 0],
                [1j, 0, 0.5, 0, 0, 1, 0, 0],
                [0, 1j, 0, 0.5, 1, 0, 0, 0],
                [0, 0, 0, 1, -0.5, 0, -1j, 0],
                [0, 0, 1, 0, 0, -0.5, 0, -1j],
                [0, 1, 0, 0, 1j, 0, -0.5, 0],
                [1, 0, 0, 0, 0, 1j, 0, -0.5],
            ],
            dtype=np.complex128,
        )
        assert np.linalg.norm(ans - obs.get_matrix().todense()) <= 1e-6  # type: ignore

    def test_get_matrix_from_general_quantum_operator(self) -> None:
        n_qubits = 3
        obs = GeneralQuantumOperator(n_qubits)
        obs.add_operator(0.5j, "Z 2")
        obs.add_operator(1.0, "X 0 X 1 X 2")
        obs.add_operator(1.0, "Y 1")
        ans = np.array(
            [
                [0.5j, 0, -1j, 0, 0, 0, 0, 1],
                [0, 0.5j, 0, -1j, 0, 0, 1, 0],
                [1j, 0, 0.5j, 0, 0, 1, 0, 0],
                [0, 1j, 0, 0.5j, 1, 0, 0, 0],
                [0, 0, 0, 1, -0.5j, 0, -1j, 0],
                [0, 0, 1, 0, 0, -0.5j, 0, -1j],
                [0, 1, 0, 0, 1j, 0, -0.5j, 0],
                [1, 0, 0, 0, 0, 1j, 0, -0.5j],
            ],
            dtype=np.complex128,
        )
        assert np.linalg.norm(ans - obs.get_matrix().todense()) <= 1e-6  # type: ignore
