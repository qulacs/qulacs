from __future__ import annotations

import qulacs_core

__all__ = [
    "create_observable_from_openfermion_file",
    "create_observable_from_openfermion_text",
    "create_split_observable",
    "from_json",
]

def create_observable_from_openfermion_file(file_path: str) -> qulacs_core.Observable:
    """
    Create GeneralQuantumOperator from openfermion file
    """

def create_observable_from_openfermion_text(text: str) -> qulacs_core.Observable:
    """
    Create GeneralQuantumOperator from openfermion text
    """

def create_split_observable(
    arg0: str,
) -> tuple[qulacs_core.Observable, qulacs_core.Observable]: ...
def from_json(json: str) -> qulacs_core.Observable:
    """
    from json string
    """
