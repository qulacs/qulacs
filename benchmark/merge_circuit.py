import qulacs
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import X

circuit = QuantumCircuit(5)
circuit2 = QuantumCircuit(5)
state = QuantumState(5)
state.set_zero_state()
circuit.add_gate(X(0))
circuit2.add_gate(X(1))
circuit.merge_circuit(circuit2)
circuit.update_quantum_state(state)
