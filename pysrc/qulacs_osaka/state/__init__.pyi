from __future__ import annotations
import qulacs_osaka_core.state
import typing
import qulacs_osaka_core

__all__ = [
    "drop_qubit",
    "inner_product",
    "partial_trace",
    "permutate_qubit",
    "tensor_product"
]


def drop_qubit(state: qulacs_osaka_core.StateVectorCpu, target: typing.List[int], projection: typing.List[int]) -> qulacs_osaka_core.StateVectorCpu:
    """
    Drop qubits from state
    """
def inner_product(state_bra: qulacs_osaka_core.StateVectorCpu, state_ket: qulacs_osaka_core.StateVectorCpu) -> complex:
    """
    Get inner product
    """
@typing.overload
def partial_trace(state: qulacs_osaka_core.DensityMatrix, target_traceout: typing.List[int]) -> qulacs_osaka_core.DensityMatrix:
    """
    Take partial trace

    Take partial trace
    """
@typing.overload
def partial_trace(state: qulacs_osaka_core.StateVectorCpu, target_traceout: typing.List[int]) -> qulacs_osaka_core.DensityMatrix:
    pass
@typing.overload
def permutate_qubit(state: qulacs_osaka_core.DensityMatrix, order: typing.List[int]) -> qulacs_osaka_core.DensityMatrix:
    """
    Permutate qubits from state

    Permutate qubits from state
    """
@typing.overload
def permutate_qubit(state: qulacs_osaka_core.StateVectorCpu, order: typing.List[int]) -> qulacs_osaka_core.StateVectorCpu:
    pass
@typing.overload
def tensor_product(state_left: qulacs_osaka_core.DensityMatrix, state_right: qulacs_osaka_core.DensityMatrix) -> qulacs_osaka_core.DensityMatrix:
    """
    Get tensor product of states

    Get tensor product of states
    """
@typing.overload
def tensor_product(state_left: qulacs_osaka_core.StateVectorCpu, state_right: qulacs_osaka_core.StateVectorCpu) -> qulacs_osaka_core.StateVectorCpu:
    pass
