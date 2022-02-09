"""FermionOperator transforms"""
from __future__ import annotations
import qulacs_osaka_core.transforms
import typing
import qulacs_osaka_core

__all__ = [
    "bravyi_kitaev",
    "jordan_wigner"
]


def bravyi_kitaev(fermion_operator: qulacs_osaka_core.FermionOperator, n_qubits: int) -> qulacs_osaka_core.Observable:
    """
    Apply the Bravyi-Kitaev transform to a FermionOperator
    """
def jordan_wigner(fermion_operator: qulacs_osaka_core.FermionOperator) -> qulacs_osaka_core.Observable:
    """
    Apply the Jordan-Wigner transform to a FermionOperator
    """
