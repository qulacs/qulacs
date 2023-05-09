import unittest

import numpy as np

import qulacs


class TestQuantumCircuit(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.dim = 2**self.n
        self.state = qulacs.QuantumState(self.n)
        self.circuit = qulacs.QuantumCircuit(self.n)

    def tearDown(self):
        del self.state
        del self.circuit

    def test_make_bell_state(self):
        self.circuit.add_H_gate(0)
        self.circuit.add_CNOT_gate(0, 1)
        self.state.set_zero_state()
        self.circuit.update_quantum_state(self.state)
        vector = self.state.get_vector()
        vector_ans = np.zeros(self.dim)
        vector_ans[0] = np.sqrt(0.5)
        vector_ans[3] = np.sqrt(0.5)
        self.assertTrue(
            ((vector - vector_ans) < 1e-10).all(), msg="check make bell state"
        )
