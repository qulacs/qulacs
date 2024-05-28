from __future__ import annotations

import typing

import numpy
import qulacs_core
import qulacs_core.gate
import scipy.sparse

__all__ = [
    "Adaptive",
    "AmplitudeDampingNoise",
    "BitFlipNoise",
    "CNOT",
    "CP",
    "CPTP",
    "CZ",
    "DenseMatrix",
    "DephasingNoise",
    "DepolarizingNoise",
    "DiagonalMatrix",
    "FREDKIN",
    "FusedSWAP",
    "H",
    "Identity",
    "IndependentXZNoise",
    "Instrument",
    "Measurement",
    "NoisyEvolution",
    "NoisyEvolution_fast",
    "P0",
    "P1",
    "ParametricPauliRotation",
    "ParametricRX",
    "ParametricRY",
    "ParametricRZ",
    "Pauli",
    "PauliRotation",
    "Probabilistic",
    "ProbabilisticInstrument",
    "RX",
    "RY",
    "RZ",
    "RandomUnitary",
    "ReversibleBoolean",
    "RotInvX",
    "RotInvY",
    "RotInvZ",
    "RotX",
    "RotY",
    "RotZ",
    "S",
    "SWAP",
    "Sdag",
    "SparseMatrix",
    "StateReflection",
    "T",
    "TOFFOLI",
    "Tdag",
    "TwoQubitDepolarizingNoise",
    "U1",
    "U2",
    "U3",
    "X",
    "Y",
    "Z",
    "add",
    "from_json",
    "merge",
    "sqrtX",
    "sqrtXdag",
    "sqrtY",
    "sqrtYdag",
    "to_matrix_gate",
]

@typing.overload
def Adaptive(
    gate: qulacs_core.QuantumGateBase,
    condition: typing.Callable[[typing.List[int]], bool],
) -> qulacs_core.QuantumGateBase:
    """
    Create adaptive gate
    """

@typing.overload
def Adaptive(
    gate: qulacs_core.QuantumGateBase,
    condition: typing.Callable[[typing.List[int], int], bool],
    id: int,
) -> qulacs_core.QuantumGateBase:
    pass

def AmplitudeDampingNoise(index: int, prob: float) -> qulacs_core.QuantumGate_CPTP:
    """
    Create amplitude damping noise
    """

def BitFlipNoise(index: int, prob: float) -> qulacs_core.QuantumGate_Probabilistic:
    """
    Create bit-flip noise
    """

def CNOT(control: int, target: int) -> qulacs_core.ClsOneControlOneTargetGate:
    """
    Create CNOT gate
    """

def CP(
    kraus_list: typing.List[qulacs_core.QuantumGateBase],
    state_normalize: bool,
    probability_normalize: bool,
    assign_zero_if_not_matched: bool,
) -> qulacs_core.QuantumGateBase:
    """
    Create completely-positive map
    """

def CPTP(
    kraus_list: typing.List[qulacs_core.QuantumGateBase],
) -> qulacs_core.QuantumGateBase:
    """
    Create completely-positive trace preserving map
    """

def CZ(control: int, target: int) -> qulacs_core.ClsOneControlOneTargetGate:
    """
    Create CZ gate
    """

@typing.overload
def DenseMatrix(index: int, matrix: numpy.ndarray) -> qulacs_core.QuantumGateMatrix:
    """
    Create dense matrix gate
    """

@typing.overload
def DenseMatrix(
    index_list: typing.List[int], matrix: numpy.ndarray
) -> qulacs_core.QuantumGateMatrix:
    pass

def DephasingNoise(index: int, prob: float) -> qulacs_core.QuantumGate_Probabilistic:
    """
    Create dephasing noise
    """

def DepolarizingNoise(index: int, prob: float) -> qulacs_core.QuantumGate_Probabilistic:
    """
    Create depolarizing noise
    """

def DiagonalMatrix(
    index_list: typing.List[int], diagonal_element: numpy.ndarray
) -> qulacs_core.QuantumGateDiagonalMatrix:
    """
    Create diagonal matrix gate
    """

def FREDKIN(control: int, target1: int, target2: int) -> qulacs_core.QuantumGateMatrix:
    """
    Create FREDKIN gate
    """

def FusedSWAP(target1: int, target2: int, block_size: int) -> ClsNpairQubitGate:
    """
    Create FusedSWAP gate
    """

def H(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create Hadamard gate
    """

def Identity(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create identity gate
    """

def IndependentXZNoise(
    index: int, prob: float
) -> qulacs_core.QuantumGate_Probabilistic:
    """
    Create independent XZ noise
    """

def Instrument(
    kraus_list: typing.List[qulacs_core.QuantumGateBase], register: int
) -> qulacs_core.QuantumGateBase:
    """
    Create instruments
    """

def Measurement(index: int, register: int) -> qulacs_core.QuantumGate_CPTP:
    """
    Create measurement gate
    """

def NoisyEvolution(
    hamiltonian: qulacs_core.Observable,
    c_ops: typing.List[qulacs_core.GeneralQuantumOperator],
    time: float,
    dt: float,
) -> qulacs_core.ClsNoisyEvolution:
    """
    Create noisy evolution
    """

def NoisyEvolution_fast(
    hamiltonian: qulacs_core.Observable,
    c_ops: typing.List[qulacs_core.GeneralQuantumOperator],
    time: float,
) -> qulacs_core.ClsNoisyEvolution_fast:
    """
    Create noisy evolution fast version
    """

def P0(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create projection gate to |0> subspace
    """

def P1(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create projection gate to |1> subspace
    """

def ParametricPauliRotation(
    index_list: typing.List[int], pauli_ids: typing.List[int], angle: float
) -> qulacs_core.QuantumGate_SingleParameter:
    """
    Create parametric multi-qubit Pauli rotation gate
    """

def ParametricRX(index: int, angle: float) -> qulacs_core.QuantumGate_SingleParameter:
    """
    Create parametric Pauli-X rotation gate
    """

def ParametricRY(index: int, angle: float) -> qulacs_core.QuantumGate_SingleParameter:
    """
    Create parametric Pauli-Y rotation gate
    """

def ParametricRZ(index: int, angle: float) -> qulacs_core.QuantumGate_SingleParameter:
    """
    Create parametric Pauli-Z rotation gate
    """

def Pauli(
    index_list: typing.List[int], pauli_ids: typing.List[int]
) -> qulacs_core.ClsPauliGate:
    """
    Create multi-qubit Pauli gate
    """

def PauliRotation(
    index_list: typing.List[int], pauli_ids: typing.List[int], angle: float
) -> qulacs_core.ClsPauliRotationGate:
    """
    Create multi-qubit Pauli rotation
    """

def Probabilistic(
    prob_list: typing.List[float], gate_list: typing.List[qulacs_core.QuantumGateBase]
) -> qulacs_core.QuantumGateBase:
    """
    Create probabilistic gate
    """

def ProbabilisticInstrument(
    prob_list: typing.List[float],
    gate_list: typing.List[qulacs_core.QuantumGateBase],
    register: int,
) -> qulacs_core.QuantumGateBase:
    """
    Create probabilistic instrument gate
    """

def RX(index: int, angle: float) -> qulacs_core.ClsOneQubitRotationGate:
    """
    Create Pauli-X rotation gate
    """

def RY(index: int, angle: float) -> qulacs_core.ClsOneQubitRotationGate:
    """
    Create Pauli-Y rotation gate
    """

def RZ(index: int, angle: float) -> qulacs_core.ClsOneQubitRotationGate:
    """
    Create Pauli-Z rotation gate
    """

@typing.overload
def RandomUnitary(index_list: typing.List[int]) -> qulacs_core.QuantumGateMatrix:
    """
    Create random unitary gate
    """

@typing.overload
def RandomUnitary(
    index_list: typing.List[int], seed: int
) -> qulacs_core.QuantumGateMatrix:
    pass

def ReversibleBoolean(
    index_list: typing.List[int], func: typing.Callable[[int, int], int]
) -> qulacs_core.ClsReversibleBooleanGate:
    """
    Create reversible boolean gate
    """

def RotInvX(index: int, angle: float) -> qulacs_core.ClsOneQubitRotationGate:
    """
    Create Pauli-X rotation gate
    """

def RotInvY(index: int, angle: float) -> qulacs_core.ClsOneQubitRotationGate:
    """
    Create Pauli-Y rotation gate
    """

def RotInvZ(index: int, angle: float) -> qulacs_core.ClsOneQubitRotationGate:
    """
    Create Pauli-Z rotation gate
    """

def RotX(index: int, angle: float) -> qulacs_core.ClsOneQubitRotationGate:
    """
    Create Pauli-X rotation gate
    """

def RotY(index: int, angle: float) -> qulacs_core.ClsOneQubitRotationGate:
    """
    Create Pauli-Y rotation gate
    """

def RotZ(index: int, angle: float) -> qulacs_core.ClsOneQubitRotationGate:
    """
    Create Pauli-Z rotation gate
    """

def S(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create pi/4-phase gate
    """

def SWAP(target1: int, target2: int) -> qulacs_core.ClsTwoQubitGate:
    """
    Create SWAP gate
    """

def Sdag(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create adjoint of pi/4-phase gate
    """

def SparseMatrix(
    index_list: typing.List[int], matrix: scipy.sparse.csc_matrix[numpy.complex128]
) -> qulacs_core.QuantumGateSparseMatrix:
    """
    Create sparse matrix gate
    """

def StateReflection(
    state: qulacs_core.QuantumState,
) -> qulacs_core.ClsStateReflectionGate:
    """
    Create state reflection gate
    """

def T(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create pi/8-phase gate
    """

def TOFFOLI(control1: int, control2: int, target: int) -> qulacs_core.QuantumGateMatrix:
    """
    Create TOFFOLI gate
    """

def Tdag(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create adjoint of pi/8-phase gate
    """

def TwoQubitDepolarizingNoise(
    index1: int, index2: int, prob: float
) -> qulacs_core.QuantumGate_Probabilistic:
    """
    Create two-qubit depolarizing noise
    """

def U1(index: int, lambda_: float) -> qulacs_core.QuantumGateMatrix:
    """
    Create QASM U1 gate
    """

def U2(index: int, phi: float, lambda_: float) -> qulacs_core.QuantumGateMatrix:
    """
    Create QASM U2 gate
    """

def U3(
    index: int, theta: float, phi: float, lambda_: float
) -> qulacs_core.QuantumGateMatrix:
    """
    Create QASM U3 gate
    """

def X(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create Pauli-X gate
    """

def Y(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create Pauli-Y gate
    """

def Z(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create Pauli-Z gate
    """

@typing.overload
def add(
    gate1: qulacs_core.QuantumGateBase, gate2: qulacs_core.QuantumGateBase
) -> qulacs_core.QuantumGateMatrix:
    """
    Add quantum gate matrices

    Add quantum gate matrices
    """

@typing.overload
def add(
    gate_list: typing.List[qulacs_core.QuantumGateBase],
) -> qulacs_core.QuantumGateMatrix:
    pass

def from_json(arg0: str) -> qulacs_core.QuantumGateBase:
    """
    from json string
    """

@typing.overload
def merge(
    gate1: qulacs_core.QuantumGateBase, gate2: qulacs_core.QuantumGateBase
) -> qulacs_core.QuantumGateMatrix:
    """
    Merge two quantum gate or gate list
    """

@typing.overload
def merge(
    gate_list: typing.List[qulacs_core.QuantumGateBase],
) -> qulacs_core.QuantumGateMatrix:
    pass

def sqrtX(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create pi/4 Pauli-X rotation gate
    """

def sqrtXdag(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create adjoint of pi/4 Pauli-X rotation gate
    """

def sqrtY(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create pi/4 Pauli-Y rotation gate
    """

def sqrtYdag(index: int) -> qulacs_core.ClsOneQubitGate:
    """
    Create adjoint of pi/4 Pauli-Y rotation gate
    """

def to_matrix_gate(gate: qulacs_core.QuantumGateBase) -> qulacs_core.QuantumGateMatrix:
    """
    Convert named gate to matrix gate
    """
