import re
import typing
from logging import NullHandler, getLogger

from qulacs import QuantumState

logger = getLogger(__name__)
logger.addHandler(NullHandler())


def convert_qulacs_state_to_strs(state: QuantumState) -> typing.List[str]:
    return state.to_string().split("\n")


def convert_strs_to_qulacs_state(strs: typing.List[str]) -> QuantumState:
    qubit = int(re.findall(r"(\d+)", strs[1])[0])
    Dim = 2**qubit
    state_vec: typing.List[complex] = [0.0 + 0.0j] * Dim
    for i in range(Dim):
        ary = re.findall(r"[-+]?\d*\.?\d+|[-+]?\d+", strs[4 + i])
        state_vec[i] = float(ary[0]) + float(ary[1]) * 1.0j
    state = QuantumState(qubit)
    state.load(state_vec)
    return state
