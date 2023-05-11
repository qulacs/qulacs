import numpy as np
import pytest
from pyparsing import Generator

import qulacs

# `pytestmark` is a variable name determined by pytest.
# If `pytestmark` is defined as skip mark, all tests in this file will be skipped.
pytestmark = pytest.mark.skipif(
    not qulacs.check_build_for_mpi(),
    reason="To use multi-cpu, qulacs built for mpi is required.",
)


class TestQuantumState:
    @pytest.fixture
    def init_state(self, init_mpi) -> Generator[None, None, None]:
        # from fixture in conftest.py
        multicpu, mpicomm = init_mpi

        self.n = 6
        self.state = qulacs.QuantumState(self.n)
        self.state_multi = qulacs.QuantumState(self.n, True)
        self.dim_all = 2**self.n
        self.dim = 2**self.n
        if multicpu:
            self.mpirank = mpicomm.Get_rank()  # type: ignore
            self.mpisize = mpicomm.Get_size()  # type: ignore
        else:
            self.mpirank = 0
            self.mpisize = 1
        if self.state_multi.get_device_name() == "multi-cpu":
            self.dim //= self.mpisize

        yield
        del self.state
        del self.state_multi

    def test_state_dim(self, init_state) -> None:
        vector = self.state_multi.get_vector()
        assert len(vector) == self.dim, "check vector size"

    def test_zero_state(self, init_state) -> None:
        self.state_multi.set_zero_state()
        vector = self.state_multi.get_vector()
        vector_ans = np.zeros(self.dim)
        if self.state_multi.get_device_name() == "cpu" or self.mpirank == 0:
            vector_ans[0] = 1.0
        assert ((vector - vector_ans) < 1e-10).all(), "check set_zero_state"

    def test_comp_basis(self, init_state) -> None:
        pos = 0b010100
        self.state_multi.set_computational_basis(pos)
        vector = self.state_multi.get_vector()
        vector_ans = np.zeros(self.dim)
        if (
            self.state_multi.get_device_name() == "cpu"
            or self.mpirank == pos // self.dim
        ):
            vector_ans[pos % self.dim] = 1.0
        assert ((vector - vector_ans) < 1e-10).all(), "check set_computational_basis"
