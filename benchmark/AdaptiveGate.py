from qulacs import QuantumState
from qulacs.gate import Adaptive, X

# OLD
adaptive = Adaptive(X(0), lambda a: (a[2] == 1))
s = QuantumState(1)
s.set_computational_basis(0)
s.set_classical_value(2, 1)
adaptive.update_quantum_state(s)
print(s)
s.set_classical_value(2, 0)
adaptive.update_quantum_state(s)
print(s)

# NEW
adaptive = Adaptive(X(0), lambda a, b: (a[b] == 1), 2)
s = QuantumState(1)
s.set_computational_basis(0)
s.set_classical_value(2, 1)
adaptive.update_quantum_state(s)
print(s)
s.set_classical_value(2, 0)
adaptive.update_quantum_state(s)
print(s)
