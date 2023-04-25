from qulacs import QuantumState
from qulacs.gate import CNOT, RY, H, X, Z
from qulacs.visualizer import (
    show_amplitude,
    show_blochsphere,
    show_probability,
)


def test_amp_pro():
    """適当な量子状態をつくって、棒グラフを表示するテスト用関数。"""
    n = 3
    state = QuantumState(n)
    # state.set_Haar_random_state()
    show_amplitude(state)
    show_probability(state)
    X(0).update_quantum_state(state)
    show_amplitude(state)
    show_probability(state)
    H(0).update_quantum_state(state)
    show_amplitude(state)
    show_probability(state)
    Z(1).update_quantum_state(state)
    show_amplitude(state)
    show_probability(state)
    CNOT(0, 1).update_quantum_state(state)
    show_amplitude(state)
    show_probability(state)


def test_bloch():
    """適当な量子状態をつくって、ブロッホ球を表示するテスト用関数。"""
    n = 3
    state = QuantumState(n)
    state.set_computational_basis(0b000)
    H(0).update_quantum_state(state)
    show_blochsphere(state, 0)
    RY(0, 0.1).update_quantum_state(state)
    show_blochsphere(state, 0)
    CNOT(0, 1).update_quantum_state(state)
    show_blochsphere(state, 0)
    show_blochsphere(state, 1)
    show_blochsphere(state, 2)
