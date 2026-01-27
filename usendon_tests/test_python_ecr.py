from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import Y,CNOT,merge, ECR

state = QuantumState(3)
state.set_Haar_random_state()

circuit = QuantumCircuit(3)
circuit.add_ECR_gate(0,1)
circuit.update_quantum_state(state)

observable = Observable(3)
observable.add_operator(2.0, "X 2 Y 1 Z 0")
observable.add_operator(-3.0, "Z 2")
value = observable.get_expectation_value(state)
print(value)