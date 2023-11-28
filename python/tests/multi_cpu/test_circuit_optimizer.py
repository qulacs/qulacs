from typing import Generator

import numpy as np
import pytest

import qulacs
from qulacs import QuantumCircuit, QuantumState
from qulacs.circuit import QuantumCircuitOptimizer

# `pytestmark` is a variable name determined by pytest.
# If `pytestmark` is defined as skip mark, all tests in this file will be skipped.
pytestmark = pytest.mark.skipif(
    not qulacs.check_build_for_mpi(),
    reason="To use multi-cpu, qulacs built for mpi is required.",
)


class TestQuantumCircuitOptimizer:
    @pytest.fixture
    def init_circuit(self, init_mpi) -> Generator[None, None, None]:
        # from fixture in conftest.py
        multicpu, mpicomm = init_mpi

        self.n = 6
        self.dim = 2**self.n
        self.dim_all = self.dim
        self.state = QuantumState(self.n)
        self.state_multi = QuantumState(self.n, True)
        self.circuit = QuantumCircuit(self.n)
        if multicpu:
            self.mpirank = mpicomm.Get_rank()  # type: ignore
            self.mpisize = mpicomm.Get_size()  # type: ignore
        else:
            self.mpirank = 0
            self.mpisize = 1
        if self.state_multi.get_device_name() == "multi-cpu":
            self.dim //= self.mpisize
        np.random.seed(seed=0)

        yield
        del self.state
        del self.state_multi
        del self.circuit

    @pytest.mark.parametrize("blocksize", [None, 0, 2])
    @pytest.mark.parametrize("swap_level", [0, 1, 2])
    @pytest.mark.usefixtures("init_circuit")
    def test_circuit_optimizer(self, blocksize, swap_level) -> None:
        # qulacs benchmark circuit with depth = 0
        for i in range(self.n):
            self.circuit.add_RX_gate(i, np.random.rand())
            self.circuit.add_RZ_gate(i, np.random.rand())
        for i in range(self.n):
            self.circuit.add_CNOT_gate(i, (i + 1) % self.n)
        for i in range(self.n):
            self.circuit.add_RZ_gate(i, np.random.rand())
            self.circuit.add_RX_gate(i, np.random.rand())

        # reference
        self.state.set_zero_state()
        self.circuit.update_quantum_state(self.state)
        vector = self.state.get_vector()

        qco = QuantumCircuitOptimizer()
        if blocksize is None:
            qco.optimize_light(self.circuit, swap_level)
        else:
            qco.optimize(self.circuit, blocksize, swap_level)

        self.state_multi.set_zero_state()
        self.circuit.update_quantum_state(self.state_multi)
        vector_multi = self.state_multi.get_vector()

        part_vector = vector[self.dim * self.mpirank : self.dim * (self.mpirank + 1)]
        assert ((vector_multi - part_vector) < 1e-10).all(), "check circuit optimizer"

    @pytest.mark.usefixtures("init_circuit")
    def test_circuit_optimizer_with_mpisize(self) -> None:
        for i in range(self.n):
            self.circuit.add_RX_gate(i, np.random.rand())
            self.circuit.add_RZ_gate(i, np.random.rand())

        qco = QuantumCircuitOptimizer(mpi_size=4)
        qco.optimize(self.circuit, 0, 2)

        is_fusedswap = [
            self.circuit.get_gate(i).get_name() == "FusedSWAP"
            for i in range(self.circuit.get_gate_count())
        ]

        assert any(is_fusedswap), "check circuit optimizer with mpi_size"
