from __future__ import annotations
import qulacs_core.state
import typing
import qulacs_core

__all__ = [
    "drop_qubit",
    "inner_product",
    "partial_trace",
    "permutate_qubit",
    "tensor_product"
]


def drop_qubit(state: qulacs_core.QuantumState, target: typing.List[int], projection: typing.List[int]) -> qulacs_core.QuantumState:
    """
    Drop qubits from state
    """
def inner_product(state_bra: qulacs_core.QuantumState, state_ket: qulacs_core.QuantumState) -> complex:
    """
    Get inner product
    """
@typing.overload
def partial_trace(state: qulacs_core.DensityMatrix, target_traceout: typing.List[int]) -> qulacs_core.DensityMatrix:
    """
    Take partial trace

    Take partial trace
    """
@typing.overload
def partial_trace(state: qulacs_core.QuantumState, target_traceout: typing.List[int]) -> qulacs_core.DensityMatrix:
    pass
@typing.overload
def permutate_qubit(state: qulacs_core.DensityMatrix, order: typing.List[int]) -> qulacs_core.DensityMatrix:
    """
    Permutate qubits from state

    Permutate qubits from state
    """
@typing.overload
def permutate_qubit(state: qulacs_core.QuantumState, order: typing.List[int]) -> qulacs_core.QuantumState:
    pass
@typing.overload
def tensor_product(state_left: qulacs_core.DensityMatrix, state_right: qulacs_core.DensityMatrix) -> qulacs_core.DensityMatrix:
    """
    Get tensor product of states

    Get tensor product of states
    """
@typing.overload
def tensor_product(state_left: qulacs_core.QuantumState, state_right: qulacs_core.QuantumState) -> qulacs_core.QuantumState:
    pass
