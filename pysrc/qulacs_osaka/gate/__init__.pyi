from __future__ import annotations
import qulacs_osaka_core.gate
import typing
import numpy
import qulacs_osaka_core
import scipy.sparse
_Shape = typing.Tuple[int, ...]

__all__ = [
    "AmplitudeDampingNoise",
    "BitFlipNoise",
    "CNOT",
    "CPTP",
    "CZ",
    "DenseMatrix",
    "DephasingNoise",
    "DepolarizingNoise",
    "DiagonalMatrix",
    "FREDKIN",
    "H",
    "Identity",
    "IndependentXZNoise",
    "Instrument",
    "Measurement",
    "P0",
    "P1",
    "Pauli",
    "PauliRotation",
    "Probabilistic",
    "RX",
    "RY",
    "RZ",
    "RandomUnitary",
    "S",
    "SWAP",
    "Sdag",
    "SparseMatrix",
    "T",
    "TOFFOLI",
    "Tdag",
    "TwoQubitDepolarizingNoise",
    "X",
    "Y",
    "Z",
    "sqrtX",
    "sqrtXdag",
    "sqrtY",
    "sqrtYdag"
]


def AmplitudeDampingNoise(index: int, prob: float) -> qulacs_osaka_core.QuantumGateWrapped:
    """
    Create amplitude damping noise
    """
def BitFlipNoise(index: int, prob: float) -> qulacs_osaka_core.QuantumGateWrapped:
    """
    Create bit-flip noise
    """
def CNOT(control: int, target: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create CNOT gate
    """
def CPTP(kraus_list: typing.List[qulacs_osaka_core.QuantumGateBase], register_name: str = '') -> qulacs_osaka_core.QuantumGateWrapped:
    """
    Create completely-positive trace preserving map
    """
def CZ(control: int, target: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create CZ gate
    """
@typing.overload
def DenseMatrix(index: int, matrix: numpy.ndarray[numpy.complex128, _Shape[m, n]]) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create dense matrix gate

    Create dense matrix gate
    """
@typing.overload
def DenseMatrix(index_list: typing.List[int], matrix: numpy.ndarray[numpy.complex128, _Shape[m, n]]) -> qulacs_osaka_core.QuantumGateBasic:
    pass
def DephasingNoise(index: int, prob: float) -> qulacs_osaka_core.QuantumGateWrapped:
    """
    Create dephasing noise
    """
def DepolarizingNoise(index: int, prob: float) -> qulacs_osaka_core.QuantumGateWrapped:
    """
    Create depolarizing noise
    """
def DiagonalMatrix(index_list: typing.List[int], diagonal_element: numpy.ndarray[numpy.complex128, _Shape[m, 1]]) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create diagonal matrix gate
    """
def FREDKIN(control: int, target1: int, target2: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create FREDKIN gate
    """
def H(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create Hadamard gate
    """
def Identity(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create identity gate
    """
def IndependentXZNoise(index: int, prob: float) -> qulacs_osaka_core.QuantumGateWrapped:
    """
    Create independent XZ noise
    """
def Instrument(kraus_list: typing.List[qulacs_osaka_core.QuantumGateBase], register_name: str) -> qulacs_osaka_core.QuantumGateWrapped:
    """
    Create instruments
    """
def Measurement(index: int, register: str) -> qulacs_osaka_core.QuantumGateWrapped:
    """
    Create measurement gate
    """
def P0(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create projection gate to |0> subspace
    """
def P1(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create projection gate to |1> subspace
    """
def Pauli(index_list: typing.List[int], pauli_ids: typing.List[int]) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create multi-qubit Pauli gate
    """
def PauliRotation(index_list: typing.List[int], pauli_ids: typing.List[int], angle: float) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create multi-qubit Pauli rotation
    """
def Probabilistic(gate_list: typing.List[qulacs_osaka_core.QuantumGateBase], prob_list: typing.List[float], register_name: str = '') -> qulacs_osaka_core.QuantumGateWrapped:
    """
    Create probabilistic gate
    """
def RX(index: int, angle: float) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create Pauli-X rotation gate
    """
def RY(index: int, angle: float) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create Pauli-Y rotation gate
    """
def RZ(index: int, angle: float) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create Pauli-Z rotation gate
    """
def RandomUnitary(index_list: typing.List[int], seed: int = -1) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create random unitary gate
    """
def S(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create pi/4-phase gate
    """
def SWAP(target1: int, target2: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create SWAP gate
    """
def Sdag(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create adjoint of pi/4-phase gate
    """
def SparseMatrix(index_list: typing.List[int], matrix: scipy.sparse.csc_matrix[numpy.complex128]) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create sparse matrix gate
    """
def T(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create pi/8-phase gate
    """
def TOFFOLI(control1: int, control2: int, target: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create TOFFOLI gate
    """
def Tdag(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create adjoint of pi/8-phase gate
    """
def TwoQubitDepolarizingNoise(index1: int, index2: int, prob: float) -> qulacs_osaka_core.QuantumGateWrapped:
    """
    Create two-qubit depolarizing noise
    """
def X(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create Pauli-X gate
    """
def Y(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create Pauli-Y gate
    """
def Z(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create Pauli-Z gate
    """
def sqrtX(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create pi/4 Pauli-X rotation gate
    """
def sqrtXdag(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create adjoint of pi/4 Pauli-X rotation gate
    """
def sqrtY(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create pi/4 Pauli-Y rotation gate
    """
def sqrtYdag(index: int) -> qulacs_osaka_core.QuantumGateBasic:
    """
    Create adjoint of pi/4 Pauli-Y rotation gate
    """
