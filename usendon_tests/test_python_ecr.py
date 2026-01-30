from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import Y,CNOT,merge, ECR


""" state = QuantumState(3)
state.set_Haar_random_state()

circuit = QuantumCircuit(3)
circuit.add_ECR_gate(0,1)
circuit.update_quantum_state(state)

observable = Observable(3)
observable.add_operator(2.0, "X 2 Y 1 Z 0")
observable.add_operator(-3.0, "Z 2")
value = observable.get_expectation_value(state)
print(value) """

from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import ECR
import numpy as np

# Crear estado
state = QuantumState(5)

# Get the state vector
vec = state.get_vector()
print(type(vec), vec.dtype)
print(vec)

# Set the state vector
myvec = np.array([0.0101105955 + 0.0167271371j,
                0.1585987468 + 0.0808763846j,
                0.2820801376 + 0.0480186846j,
                -0.0240327857 - 0.0574569032j,
                0.0858650333 + 0.2115784928j,
                0.1657284343 - 0.0831781034j,
                -0.1367750715 - 0.1365907697j,
                -0.0882971706 + 0.0984543525j,
                0.1204164159 - 0.0667672630j,
                0.0914460776 - 0.1067782961j,
                0.0363543846 + 0.0023833030j,
                -0.0767846302 + 0.1234282081j,
                0.0066363518 - 0.0148286465j,
                0.0129140401 - 0.0271140316j,
                -0.0225181255 + 0.2397009974j,
                -0.0890724897 + 0.1518390685j,
                0.1228845220 + 0.0055693079j,
                0.1737948010 + 0.1825285070j,
                -0.0788327840 - 0.0068239610j,
                0.1435911529 + 0.1538036339j,
                -0.0543893547 - 0.1873199665j,
                0.1588434092 - 0.0838706501j,
                -0.0560615395 - 0.0229029912j,
                -0.2549819852 + 0.0476732944j,
                0.0513528643 - 0.2798395514j,
                -0.1730591396 + 0.0146931346j,
                0.0491209904 - 0.0320089085j,
                0.2103362291 - 0.0238378796j,
                0.0636509085 - 0.0305138954j,
                -0.0996959250 - 0.3243868490j,
                -0.1503251987 + 0.1018871130j,
                0.0717523446 - 0.0374508447j
])

state.load(myvec)
print(state)

# Crear circuito
circuit = QuantumCircuit(5)
circuit.add_ECR_gate(3,4)

# Aplicar circuito
circuit.update_quantum_state(state)

print("Estado final:")
print(state.get_vector())


