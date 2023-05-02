import numpy as np
import pytest
from scipy.sparse import csr_matrix

from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import (
    CPTP,
    DenseMatrix,
    Instrument,
    Measurement,
    Probabilistic,
)

nqubits_list = range(4, 26)


def bench_gate(benchmark, nqubits, g):
    st = QuantumState(nqubits)
    benchmark(g.update_quantum_state, st)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Dense1Q(benchmark, nqubits):
    benchmark.group = "DenseMatrix1Q"
    bench_gate(benchmark, nqubits, DenseMatrix([3], np.eye(2)))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_CPTP1Q(benchmark, nqubits):
    benchmark.group = "CPTP1Q"
    g = DenseMatrix(
        [
            3,
        ],
        np.eye(2) / np.sqrt(2),
    )
    cptp = CPTP([g, g])
    bench_gate(benchmark, nqubits, cptp)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Instrument1Q(benchmark, nqubits):
    benchmark.group = "Instrument1Q"
    bench_gate(benchmark, nqubits, Measurement(3, 0))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Probabilistic1Q(benchmark, nqubits):
    benchmark.group = "Probabilistic1Q"
    g = DenseMatrix(
        [
            3,
        ],
        np.eye(2) / np.sqrt(2),
    )
    gate = Probabilistic([0.5, 0.5], [g, g])
    bench_gate(benchmark, nqubits, gate)
