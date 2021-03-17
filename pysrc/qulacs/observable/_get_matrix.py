from qulacs_core.observable import *
from scipy.sparse import csr_matrix, kron
import numpy as np

sigmaz = csr_matrix([[1, 0], [0, -1]])
sigmay = csr_matrix([[0, -1j], [1j, 0]])
sigmax = csr_matrix([[0, 1], [1, 0]])
sigmai = csr_matrix([[1, 0], [0, 1]])

sigma_list = [sigmai, sigmax, sigmay, sigmaz]


def _kron_n(*ops):
    """
    takes tensor product of given scipy matrix
    Args:
        ops (:class:`list`) 
    """
    if len(ops) == 2:
        return kron(ops[0], ops[1])
    else:
        return kron(_kron_n(*ops[:-1]), ops[-1])


def _get_matrix(obs):
    """
    returns matrix of an observable
    Args:
        obs (qulacs_core.Observable)
    Return:
        scipy.sparse.csr_matrix
    """
    n_terms = obs.get_term_count()
    n_qubits = obs.get_qubit_count()
    result = csr_matrix((2**n_qubits, 2**n_qubits), dtype=np.complex128)
    for i in range(n_terms):
        pauli = obs.get_term(i)
        pauli_id_list = pauli.get_pauli_id_list()
        pauli_target_list = pauli.get_index_list()
        pauli_string = [sigmai for q in range(n_qubits)]
        for j, target in enumerate(pauli_target_list):
            pauli_string[target] = sigma_list[pauli_id_list[j]]
        result += pauli.get_coef()*_kron_n(*(pauli_string[::-1]))
    return result
