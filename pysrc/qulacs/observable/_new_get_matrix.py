from typing import List

import numpy as np
from qulacs_core import Observable
from scipy.sparse import csr_matrix, kron


def _one_step_execute(_obs: List[Observable]) -> List[Observable]:
    _new_obs = []
    for i in range(0, len(_obs), 2):
        _op = kron(_obs[i], _obs[i + 1], format="csr")
        _new_obs.append(_op)
    return _new_obs


def _return_tensor_product(_obs: List[Observable]) -> csr_matrix:
    while len(_obs) != 1:
        if len(_obs) % 2 == 0:
            _obs = _one_step_execute(_obs=_obs)
        else:
            _obs[-2] = kron(_obs[-2], _obs[-1], format="csr")
            del _obs[-1]
    assert len(_obs) == 1, "raise unrecognised error."
    return _obs[0]


def _new_get_matrix(obs: Observable) -> csr_matrix:
    Z = csr_matrix([[1, 0], [0, -1]], dtype=np.complex128)
    Y = csr_matrix([[0, -1j], [1j, 0]], dtype=np.complex128)
    X = csr_matrix([[0, 1], [1, 0]], dtype=np.complex128)
    I = csr_matrix([[1, 0], [0, 1]], dtype=np.complex128)
    pauli_matrix_list = [I, X, Y, Z]

    n_terms = obs.get_term_count()
    n_qubits = obs.get_qubit_count()

    hamiltonian_matrix = csr_matrix((2**n_qubits, 2**n_qubits), dtype=np.complex128)
    for i in range(n_terms):
        pauli = obs.get_term(i)
        pauli_id_list = pauli.get_pauli_id_list()
        pauli_target_list = pauli.get_index_list()

        init_hamiltonian_pauli_matrix_list = [
            I for _ in range(n_qubits)
        ]  # initialize matrix_list I
        for j, target in enumerate(pauli_target_list):
            init_hamiltonian_pauli_matrix_list[target] = pauli_matrix_list[
                pauli_id_list[j]
            ]  # ex) [X,X,I,I]
        hamiltonian_matrix += pauli.get_coef() * _return_tensor_product(
            init_hamiltonian_pauli_matrix_list
        )
    return hamiltonian_matrix
