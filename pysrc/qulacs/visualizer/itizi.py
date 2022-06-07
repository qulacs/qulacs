
from qulacs import QuantumGateMatrix
from qulacs import QuantumState

import numpy as np

gate = QuantumGateMatrix([0, 1], [[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])

gateB=gate.copy()

gateB.add_control_qubit(2,1)
state = QuantumState(3)

gateB.update_quantum_state(state)
