from __future__ import annotations
import qulacs_core.observable
import typing
import qulacs_core

__all__ = [
    "create_observable_from_openfermion_file",
    "create_observable_from_openfermion_text",
    "create_split_observable"
]


def create_observable_from_openfermion_file(file_path: str) -> qulacs_core.Observable:
    """
    Create GeneralQuantumOperator from openfermion file
    """
def create_observable_from_openfermion_text(text: str) -> qulacs_core.Observable:
    """
    Create GeneralQuantumOperator from openfermion text
    """
def create_split_observable(arg0: str) -> typing.Tuple[qulacs_core.Observable, qulacs_core.Observable]:
    pass
