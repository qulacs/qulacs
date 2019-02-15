
# set library dir
import sys
for ind in range(1,len(sys.argv)):
    sys.path.append(sys.argv[ind])
sys.argv = sys.argv[:1]

import numpy as np
import unittest
import qulacs

class TestQuantumState(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.dim = 2**self.n
        self.state = qulacs.QuantumState(self.n)

    def tearDown(self):
        del self.state

    def test_state_dim(self):
        vector = self.state.get_vector()
        self.assertEqual(len(vector),self.dim, msg = "check vector size")

    def test_zero_state(self):
        self.state.set_zero_state()
        vector = self.state.get_vector()
        vector_ans = np.zeros(self.dim)
        vector_ans[0]=1.
        self.assertTrue(((vector-vector_ans)<1e-10).all(), msg = "check set_zero_state")

    def test_comp_basis(self):
        pos = 0b0101
        self.state.set_computational_basis(pos)
        vector = self.state.get_vector()
        vector_ans = np.zeros(self.dim)
        vector_ans[pos]=1.
        self.assertTrue(((vector-vector_ans)<1e-10).all(), msg = "check set_computational_basis")

class TestQuantumCircuit(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.dim = 2**self.n
        self.state = qulacs.QuantumState(self.n)
        self.circuit = qulacs.QuantumCircuit(self.n)

    def tearDown(self):
        del self.state
        del self.circuit

    def test_make_bell_state(self):
        self.circuit.add_H_gate(0)
        self.circuit.add_CNOT_gate(0,1)
        self.state.set_zero_state()
        self.circuit.update_quantum_state(self.state)
        vector = self.state.get_vector()
        vector_ans = np.zeros(self.dim)
        vector_ans[0] = np.sqrt(0.5)
        vector_ans[3] = np.sqrt(0.5)
        self.assertTrue(((vector-vector_ans)<1e-10).all(), msg = "check make bell state")

class TestPointerHandling(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_pointer_del(self):
        from qulacs import QuantumCircuit
        from qulacs.gate import X
        qc = QuantumCircuit(1)
        gate = X(0)
        qc.add_gate(gate)
        del gate
        del qc

    def test_internal_return_value_of_get_gate_is_valid(self):
        from qulacs import QuantumCircuit, QuantumState
        def func():
            def copy_circuit(c):
                ret = QuantumCircuit(2)
                for i in range(c.get_gate_count()):
                    gate = c.get_gate(i)
                    ret.add_gate(gate)
                return ret

            circuit = QuantumCircuit(2)
            circuit.add_X_gate(0)
            circuit.add_Y_gate(1)
            copied = copy_circuit(circuit)
            return copied

        def func2():
            qs = QuantumState(2)
            circuit = func()
            circuit.update_quantum_state(qs)

        func2()

    def test_add_same_gate_multiple_time(self):
        from qulacs import QuantumCircuit, QuantumState
        from qulacs.gate import X, DepolarizingNoise, DephasingNoise, Probabilistic, RX
        state = QuantumState(1)
        circuit = QuantumCircuit(1)
        noise = DepolarizingNoise(0,0)
        circuit.add_gate(noise)
        circuit.add_gate(noise.copy())
        circuit.add_gate(DephasingNoise(0,0))
        circuit.add_gate(Probabilistic([0.1],[RX(0,0)]))
        gate = RX(0,0)
        circuit.add_gate(gate)
        circuit.add_gate(gate)
        circuit.add_gate(gate)
        circuit.add_gate(gate)
        circuit.add_gate(gate)
        del gate
        circuit.update_quantum_state(state)
        circuit.update_quantum_state(state)
        circuit.update_quantum_state(state)
        circuit.update_quantum_state(state)
        circuit.update_quantum_state(state)
        circuit.update_quantum_state(state)
        circuit.update_quantum_state(state)
        circuit.to_string()
        del circuit
        del state
    
    def test_observable(self):
        from qulacs import Observable
        obs = Observable(1)
        obs.add_operator(1.0, "X 0")
        term = obs.get_term(0)

    def test_add_gate(self):
        from qulacs import QuantumCircuit
        from qulacs.gate import X
        circuit = QuantumCircuit(1)
        gate = X(0)
        circuit.add_gate(gate)
        del gate
        s = circuit.to_string()
        del circuit

    def test_add_gate_in_parametric_circuit(self):
        from qulacs import ParametricQuantumCircuit
        from qulacs.gate import X
        circuit = ParametricQuantumCircuit(1)
        gate = X(0)
        circuit.add_gate(gate)
        del gate
        s = circuit.to_string()
        del circuit
        
    def test_state_reflection(self):
        from qulacs import QuantumState
        from qulacs.gate import StateReflection
        n = 5
        s1 = QuantumState(n)
        def gen_gate():
            s2 = QuantumState(n)
            gate = StateReflection(s2)
            del s2
            return gate
        gate = gen_gate()
        gate.update_quantum_state(s1)
        del gate
        del s1

    def test_sparse_matrix(self):
        
        from qulacs import QuantumState
        from qulacs.gate import SparseMatrix
        from scipy.sparse import lil_matrix
        n = 5
        state = QuantumState(n)
        matrix = lil_matrix( (4,4) , dtype = np.complex128)
        matrix[0,0] = 1 + 1.j
        matrix[1,1] = 1. + 1.j
        gate = SparseMatrix([0,1], matrix)
        gate.update_quantum_state(state)
        del gate
        del state
        

if __name__ == "__main__":
    unittest.main()