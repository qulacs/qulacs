from __future__ import annotations

import typing

import qulacs_core
import qulacs_core.quantum_operator

__all__ = [
    "create_quantum_operator_from_openfermion_file",
    "create_quantum_operator_from_openfermion_text",
    "create_split_quantum_operator",
    "from_json",
]

def create_quantum_operator_from_openfermion_file(
    arg0: str,
) -> qulacs_core.GeneralQuantumOperator:
    pass

def create_quantum_operator_from_openfermion_text(
    arg0: str,
) -> qulacs_core.GeneralQuantumOperator:
    pass

def create_split_quantum_operator(
    arg0: str,
) -> typing.Tuple[
    qulacs_core.GeneralQuantumOperator, qulacs_core.GeneralQuantumOperator
]:
    pass

def from_json(json: str) -> qulacs_core.GeneralQuantumOperator:
    """
    from json string
    """
