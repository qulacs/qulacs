from typing import Generator

import numpy as np
import pytest

from qulacs import QuantumState


class TestQuantumState:
    @pytest.fixture
    def init_state(self) -> Generator[None, None, None]:
        self.n = 4
        self.dim = 2**self.n
        self.state = QuantumState(self.n)

        yield
        del self.state

    def test_state_dim(self, init_state) -> None:
        vector = self.state.get_vector()
        assert len(vector) == self.dim, "check vector size"

    def test_zero_state(self, init_state) -> None:
        self.state.set_zero_state()
        vector = self.state.get_vector()
        vector_ans = np.zeros(self.dim)
        vector_ans[0] = 1.0
        assert ((vector - vector_ans) < 1e-10).all(), "check set_zero_state"

    def test_comp_basis(self, init_state) -> None:
        pos = 0b0101
        self.state.set_computational_basis(pos)
        vector = self.state.get_vector()
        vector_ans = np.zeros(self.dim)
        vector_ans[pos] = 1.0
        assert ((vector - vector_ans) < 1e-10).all(), "check set_computational_basis"
