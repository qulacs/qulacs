from typing import Generator

import numpy as np
import pytest

from qulacs import QuantumCircuit, QuantumState


class TestQuantumCircuit:
    @pytest.fixture
    def init_circuit(self) -> Generator[None, None, None]:
        self.n = 4
        self.dim = 2**self.n
        self.state = QuantumState(self.n)
        self.circuit = QuantumCircuit(self.n)

        yield
        del self.state
        del self.circuit

    def test_make_bell_state(self, init_circuit) -> None:
        self.circuit.add_H_gate(0)
        self.circuit.add_CNOT_gate(0, 1)
        self.state.set_zero_state()
        self.circuit.update_quantum_state(self.state)
        vector = self.state.get_vector()
        vector_ans = np.zeros(self.dim)
        vector_ans[0] = np.sqrt(0.5)
        vector_ans[3] = np.sqrt(0.5)
        assert ((vector - vector_ans) < 1e-10).all(), "check make bell state"
