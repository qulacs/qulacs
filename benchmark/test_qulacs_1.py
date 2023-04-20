import numpy as np
import pytest
from scipy.sparse import csr_matrix

from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import (
    CNOT,
    TOFFOLI,
    DenseMatrix,
    DiagonalMatrix,
    H,
    PauliRotation,
    ReversibleBoolean,
    SparseMatrix,
    T,
    X,
)

nqubits_list = range(4, 26)


def bench_gate(benchmark, nqubits, g):
    st = QuantumState(nqubits)
    benchmark(g.update_quantum_state, st)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_X(benchmark, nqubits):
    benchmark.group = "X"
    bench_gate(benchmark, nqubits, X(3))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_CNOT(benchmark, nqubits):
    benchmark.group = "CNOT"
    bench_gate(benchmark, nqubits, CNOT(2, 3))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_TOFFOLI(benchmark, nqubits):
    benchmark.group = "TOFFOLI"
    bench_gate(benchmark, nqubits, TOFFOLI(1, 2, 3))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_CCCNOT(benchmark, nqubits):
    benchmark.group = "CCCNOT"
    g = DenseMatrix([3], [[0, 1], [1, 0]])
    g.add_control_qubit(2, 1)
    g.add_control_qubit(1, 1)
    g.add_control_qubit(0, 1)
    bench_gate(benchmark, nqubits, g)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Dense1Q(benchmark, nqubits):
    benchmark.group = "DenseMatrix1Q"
    bench_gate(benchmark, nqubits, DenseMatrix([3], np.eye(2)))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Dense2Q(benchmark, nqubits):
    benchmark.group = "DenseMatrix2Q"
    bench_gate(benchmark, nqubits, DenseMatrix([2, 3], np.eye(4)))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Dense3Q(benchmark, nqubits):
    benchmark.group = "DenseMatrix3Q"
    bench_gate(benchmark, nqubits, DenseMatrix([1, 2, 3], np.eye(8)))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Dense4Q(benchmark, nqubits):
    benchmark.group = "DenseMatrix4Q"
    bench_gate(benchmark, nqubits, DenseMatrix([0, 1, 2, 3], np.eye(16)))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Diagonal1QSp(benchmark, nqubits):
    benchmark.group = "DiagonalMatrix1QSp"
    bench_gate(benchmark, nqubits, DiagonalMatrix([0], np.ones(2)))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Diagonal1Q(benchmark, nqubits):
    benchmark.group = "DiagonalMatrix1Q"
    bench_gate(benchmark, nqubits, DiagonalMatrix([3], np.ones(2)))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Diagonal2Q(benchmark, nqubits):
    benchmark.group = "DiagonalMatrix2Q"
    bench_gate(benchmark, nqubits, DiagonalMatrix([2, 3], np.ones(4)))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Diagonal3Q(benchmark, nqubits):
    benchmark.group = "DiagonalMatrix3Q"
    bench_gate(benchmark, nqubits, DiagonalMatrix([1, 2, 3], np.ones(8)))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Diagonal4Q(benchmark, nqubits):
    benchmark.group = "DiagonalMatrix4Q"
    bench_gate(benchmark, nqubits, DiagonalMatrix([0, 1, 2, 3], np.ones(16)))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Sparse1Q(benchmark, nqubits):
    benchmark.group = "SparseMatrix1Q"
    sparse_matrix = csr_matrix(([1], ([0], [0])), shape=(2, 2), dtype=complex)
    bench_gate(benchmark, nqubits, SparseMatrix([3], sparse_matrix))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Sparse2Q(benchmark, nqubits):
    benchmark.group = "SparseMatrix2Q"
    sparse_matrix = csr_matrix(([1], ([0], [0])), shape=(4, 4), dtype=complex)
    bench_gate(benchmark, nqubits, SparseMatrix([2, 3], sparse_matrix))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Sparse3Q(benchmark, nqubits):
    benchmark.group = "SparseMatrix3Q"
    sparse_matrix = csr_matrix(([1], ([0], [0])), shape=(8, 8), dtype=complex)
    bench_gate(benchmark, nqubits, SparseMatrix([1, 2, 3], sparse_matrix))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Sparse4Q(benchmark, nqubits):
    benchmark.group = "SparseMatrix4Q"
    sparse_matrix = csr_matrix(([1], ([0], [0])), shape=(16, 16), dtype=complex)
    bench_gate(benchmark, nqubits, SparseMatrix([0, 1, 2, 3], sparse_matrix))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Permutation1Q(benchmark, nqubits):
    benchmark.group = "Permutation1Q"

    def rev(index, dim):
        return (index + 1) % dim

    bench_gate(benchmark, nqubits, ReversibleBoolean([3], rev))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Permutation2Q(benchmark, nqubits):
    benchmark.group = "Permutation2Q"

    def rev(index, dim):
        return (index + 1) % dim

    bench_gate(benchmark, nqubits, ReversibleBoolean([2, 3], rev))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Permutation3Q(benchmark, nqubits):
    benchmark.group = "Permutation3Q"

    def rev(index, dim):
        return (index + 1) % dim

    bench_gate(benchmark, nqubits, ReversibleBoolean([1, 2, 3], rev))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_Permutation4Q(benchmark, nqubits):
    benchmark.group = "Permutation4Q"

    def rev(index, dim):
        return (index + 1) % dim

    bench_gate(benchmark, nqubits, ReversibleBoolean([0, 1, 2, 3], rev))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_PauliRotation1Q(benchmark, nqubits):
    benchmark.group = "PauliRotation1Q"
    bench_gate(benchmark, nqubits, PauliRotation([3], [1], 0.1))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_PauliRotation2Q(benchmark, nqubits):
    benchmark.group = "PauliRotation2Q"
    bench_gate(benchmark, nqubits, PauliRotation([2, 3], [1, 1], 0.1))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_PauliRotation3Q(benchmark, nqubits):
    benchmark.group = "PauliRotation3Q"
    bench_gate(benchmark, nqubits, PauliRotation([1, 2, 3], [1, 1, 1], 0.1))


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_PauliRotation4Q(benchmark, nqubits):
    benchmark.group = "PauliRotation4Q"
    bench_gate(benchmark, nqubits, PauliRotation([0, 1, 2, 3], [1, 1, 1, 1], 0.1))
