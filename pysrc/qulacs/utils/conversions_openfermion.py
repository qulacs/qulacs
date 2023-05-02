import numpy as np
from qulacs_core import GeneralQuantumOperator


def convert_openfermion_op(openfermion_op, n_qubits=None):
    """convert_openfermion_op
    Args:
        openfermion_op (:class:`openfermion.ops.QubitOperator`)
        n_qubit (:class:`int`):
            if None (default), it automatically calculates the number of qubits required to represent the given operator
    Returns:
        :class:`qulacs.GeneralQuantumOperator`
    """
    if n_qubits is None:
        _n_qubits = _count_qubit_in_qubit_operator(openfermion_op)
    else:
        _n_qubits = n_qubits
    res = GeneralQuantumOperator(_n_qubits)
    for pauli_product in openfermion_op.terms:
        coef = float(np.real(openfermion_op.terms[pauli_product]))
        pauli_string = ""
        for pauli_operator in pauli_product:
            pauli_string += pauli_operator[1] + " " + str(pauli_operator[0])
            pauli_string += " "
        res.add_operator(coef, pauli_string[:-1])
    return res


def _count_qubit_in_qubit_operator(op):
    """_count_qubit_in_qubit_operator
    counts minimal number of qubits required to represent a given QubitOperator
    Args:
        openfermion_op (:class:`openfermion.ops.QubitOperator`)
    Return:
        :class: `int`
    """
    n_qubits = 0
    for pauli_product in op.terms:
        for pauli_operator in pauli_product:
            if n_qubits < pauli_operator[0]:
                n_qubits = pauli_operator[0]
    return n_qubits + 1
