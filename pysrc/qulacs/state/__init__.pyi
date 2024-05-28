from __future__ import annotations

import typing

import qulacs_core
import qulacs_core.state

__all__ = [
    "drop_qubit",
    "from_json",
    "inner_product",
    "make_mixture",
    "make_superposition",
    "partial_trace",
    "permutate_qubit",
    "tensor_product",
]

def drop_qubit(
    state: qulacs_core.QuantumState,
    target: typing.List[int],
    projection: typing.List[int],
) -> qulacs_core.QuantumState:
    """
    Drop qubits from state
    """

def from_json(json: str) -> qulacs_core.QuantumStateBase:
    """
    from json string
    """

def inner_product(
    state_bra: qulacs_core.QuantumState, state_ket: qulacs_core.QuantumState
) -> complex:
    """
    Get inner product
    """

def make_mixture(
    prob1: complex,
    state1: qulacs_core.QuantumStateBase,
    prob2: complex,
    state2: qulacs_core.QuantumStateBase,
) -> qulacs_core.DensityMatrix:
    """
    Create a mixed state
    """

def make_superposition(
    coef1: complex,
    state1: qulacs_core.QuantumState,
    coef2: complex,
    state2: qulacs_core.QuantumState,
) -> qulacs_core.QuantumState:
    """
    Create superposition of states
    """

@typing.overload
def partial_trace(
    state: qulacs_core.QuantumState, target_traceout: typing.List[int]
) -> qulacs_core.DensityMatrix:
    """
    Take partial trace
    """

@typing.overload
def partial_trace(
    state: qulacs_core.DensityMatrix, target_traceout: typing.List[int]
) -> qulacs_core.DensityMatrix:
    pass

@typing.overload
def permutate_qubit(
    state: qulacs_core.QuantumState, qubit_order: typing.List[int]
) -> qulacs_core.QuantumState:
    """
    Permutate qubits from state
    """

@typing.overload
def permutate_qubit(
    state: qulacs_core.DensityMatrix, qubit_order: typing.List[int]
) -> qulacs_core.DensityMatrix:
    pass

@typing.overload
def tensor_product(
    state_left: qulacs_core.QuantumState, state_right: qulacs_core.QuantumState
) -> qulacs_core.QuantumState:
    """
    Get tensor product of states
    """

@typing.overload
def tensor_product(
    state_left: qulacs_core.DensityMatrix, state_right: qulacs_core.DensityMatrix
) -> qulacs_core.DensityMatrix:
    pass
