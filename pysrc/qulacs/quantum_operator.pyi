from __future__ import annotations

import qulacs_core

__all__ = [
    "create_quantum_operator_from_openfermion_file",
    "create_quantum_operator_from_openfermion_text",
    "create_split_quantum_operator",
    "from_json",
]

def create_quantum_operator_from_openfermion_file(
    arg0: str,
) -> qulacs_core.GeneralQuantumOperator: ...
def create_quantum_operator_from_openfermion_text(
    arg0: str,
) -> qulacs_core.GeneralQuantumOperator: ...
def create_split_quantum_operator(
    arg0: str,
) -> tuple[qulacs_core.GeneralQuantumOperator, qulacs_core.GeneralQuantumOperator]: ...
def from_json(json: str) -> qulacs_core.GeneralQuantumOperator:
    """
    from json string
    """
