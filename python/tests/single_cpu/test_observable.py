import unittest


class TestObservable(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_matrix(self):
        import numpy as np

        from qulacs import Observable

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
        self.assertLessEqual(np.linalg.norm(ans - obs.get_matrix().todense()), 1e-6)
        from qulacs import GeneralQuantumOperator

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
        self.assertLessEqual(np.linalg.norm(ans - obs.get_matrix().todense()), 1e-6)
