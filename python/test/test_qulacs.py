
# set library dir
import qulacs
import unittest
import numpy as np
import sys
for ind in range(1, len(sys.argv)):
    sys.path.append(sys.argv[ind])
sys.argv = sys.argv[:1]


class TestQuantumState(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.dim = 2**self.n
        self.state = qulacs.QuantumState(self.n)

    def tearDown(self):
        del self.state

    def test_state_dim(self):
        vector = self.state.get_vector()
        self.assertEqual(len(vector), self.dim, msg="check vector size")

    def test_zero_state(self):
        self.state.set_zero_state()
        vector = self.state.get_vector()
        vector_ans = np.zeros(self.dim)
        vector_ans[0] = 1.
        self.assertTrue(((vector - vector_ans) < 1e-10).all(), msg="check set_zero_state")

    def test_comp_basis(self):
        pos = 0b0101
        self.state.set_computational_basis(pos)
        vector = self.state.get_vector()
        vector_ans = np.zeros(self.dim)
        vector_ans[pos] = 1.
        self.assertTrue(((vector - vector_ans) < 1e-10).all(), msg="check set_computational_basis")


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
        self.circuit.add_CNOT_gate(0, 1)
        self.state.set_zero_state()
        self.circuit.update_quantum_state(self.state)
        vector = self.state.get_vector()
        vector_ans = np.zeros(self.dim)
        vector_ans[0] = np.sqrt(0.5)
        vector_ans[3] = np.sqrt(0.5)
        self.assertTrue(((vector - vector_ans) < 1e-10).all(), msg="check make bell state")


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

    def test_circuit_add_gate(self):
        from qulacs import QuantumCircuit, QuantumState
        from qulacs.gate import Identity, X, Y, Z, H, S, Sdag, T, Tdag, sqrtX, sqrtXdag, sqrtY, sqrtYdag
        from qulacs.gate import P0, P1, U1, U2, U3, RX, RY, RZ, CNOT, CZ, SWAP, TOFFOLI, FREDKIN, Pauli, PauliRotation
        from qulacs.gate import DenseMatrix, SparseMatrix, DiagonalMatrix, RandomUnitary, ReversibleBoolean, StateReflection
        from qulacs.gate import BitFlipNoise, DephasingNoise, IndependentXZNoise, DepolarizingNoise, TwoQubitDepolarizingNoise, AmplitudeDampingNoise, Measurement
        from qulacs.gate import merge, add, to_matrix_gate, Probabilistic, CPTP, Instrument, Adaptive
        from scipy.sparse import lil_matrix
        qc = QuantumCircuit(3)
        qs = QuantumState(3)
        ref = QuantumState(3)
        sparse_mat = lil_matrix((4, 4))
        sparse_mat[0, 0] = 1
        sparse_mat[1, 1] = 1

        def func(v, d):
            return (v + 1) % d

        def adap(v):
            return True

        gates = [
            Identity(0), X(0), Y(0), Z(0), H(0), S(0), Sdag(0), T(0), Tdag(0), sqrtX(0), sqrtXdag(0), sqrtY(0), sqrtYdag(0),
            Probabilistic([0.5, 0.5], [X(0), Y(0)]), CPTP([P0(0), P1(0)]), Instrument([P0(0), P1(0)], 1), Adaptive(X(0), adap),
            CNOT(0, 1), CZ(0, 1), SWAP(0, 1), TOFFOLI(0, 1, 2), FREDKIN(0, 1, 2), Pauli([0, 1], [1, 2]), PauliRotation([0, 1], [1, 2], 0.1),
            DenseMatrix(0, np.eye(2)), DenseMatrix([0, 1], np.eye(4)), SparseMatrix([0, 1], sparse_mat),
            DiagonalMatrix([0, 1], np.ones(4)), RandomUnitary([0, 1]), ReversibleBoolean([0, 1], func), StateReflection(ref),
            BitFlipNoise(0, 0.1), DephasingNoise(0, 0.1), IndependentXZNoise(0, 0.1), DepolarizingNoise(0, 0.1), TwoQubitDepolarizingNoise(0, 1, 0.1),
            AmplitudeDampingNoise(0, 0.1), Measurement(0, 1), merge(X(0), Y(1)), add(X(0), Y(1)), to_matrix_gate(X(0)),
            P0(0), P1(0), U1(0, 0.), U2(0, 0., 0.), U3(0, 0., 0., 0.), RX(0, 0.), RY(0, 0.), RZ(0, 0.),
        ]
        gates.append(merge(gates[0], gates[1]))
        gates.append(add(gates[0], gates[1]))

        ref = None
        for gate in gates:
            qc.add_gate(gate)

        for gate in gates:
            qc.add_gate(gate)

        qc.update_quantum_state(qs)
        qc = None
        qs = None
        for gate in gates:
            gate = None

        gates = None
        parametric_gates = None

    def test_circuit_add_parametric_gate(self):
        from qulacs import ParametricQuantumCircuit, QuantumState
        from qulacs.gate import Identity, X, Y, Z, H, S, Sdag, T, Tdag, sqrtX, sqrtXdag, sqrtY, sqrtYdag
        from qulacs.gate import P0, P1, U1, U2, U3, RX, RY, RZ, CNOT, CZ, SWAP, TOFFOLI, FREDKIN, Pauli, PauliRotation
        from qulacs.gate import DenseMatrix, SparseMatrix, DiagonalMatrix, RandomUnitary, ReversibleBoolean, StateReflection
        from qulacs.gate import BitFlipNoise, DephasingNoise, IndependentXZNoise, DepolarizingNoise, TwoQubitDepolarizingNoise, AmplitudeDampingNoise, Measurement
        from qulacs.gate import merge, add, to_matrix_gate, Probabilistic, CPTP, Instrument, Adaptive
        from qulacs.gate import ParametricRX, ParametricRY, ParametricRZ, ParametricPauliRotation
        from scipy.sparse import lil_matrix
        qc = ParametricQuantumCircuit(3)
        qs = QuantumState(3)
        ref = QuantumState(3)
        sparse_mat = lil_matrix((4, 4))
        sparse_mat[0, 0] = 1
        sparse_mat[1, 1] = 1

        def func(v, d):
            return (v + 1) % d

        def adap(v):
            return True

        gates = [
            Identity(0), X(0), Y(0), Z(0), H(0), S(0), Sdag(0), T(0), Tdag(0), sqrtX(0), sqrtXdag(0), sqrtY(0), sqrtYdag(0),
            Probabilistic([0.5, 0.5], [X(0), Y(0)]), CPTP([P0(0), P1(0)]), Instrument([P0(0), P1(0)], 1), Adaptive(X(0), adap),
            CNOT(0, 1), CZ(0, 1), SWAP(0, 1), TOFFOLI(0, 1, 2), FREDKIN(0, 1, 2), Pauli([0, 1], [1, 2]), PauliRotation([0, 1], [1, 2], 0.1),
            DenseMatrix(0, np.eye(2)), DenseMatrix([0, 1], np.eye(4)), SparseMatrix([0, 1], sparse_mat),
            DiagonalMatrix([0, 1], np.ones(4)), RandomUnitary([0, 1]), ReversibleBoolean([0, 1], func), StateReflection(ref),
            BitFlipNoise(0, 0.1), DephasingNoise(0, 0.1), IndependentXZNoise(0, 0.1), DepolarizingNoise(0, 0.1), TwoQubitDepolarizingNoise(0, 1, 0.1),
            AmplitudeDampingNoise(0, 0.1), Measurement(0, 1), merge(X(0), Y(1)), add(X(0), Y(1)), to_matrix_gate(X(0)),
            P0(0), P1(0), U1(0, 0.), U2(0, 0., 0.), U3(0, 0., 0., 0.), RX(0, 0.), RY(0, 0.), RZ(0, 0.),
        ]

        gates.append(merge(gates[0], gates[1]))
        gates.append(add(gates[0], gates[1]))

        parametric_gates = [
            ParametricRX(0, 0.1), ParametricRY(0, 0.1), ParametricRZ(0, 0.1), ParametricPauliRotation([0, 1], [1, 1], 0.1)
        ]

        ref = None
        for gate in gates:
            qc.add_gate(gate)

        for gate in gates:
            qc.add_gate(gate)

        for pgate in parametric_gates:
            qc.add_parametric_gate(pgate)

        for pgate in parametric_gates:
            qc.add_parametric_gate(pgate)

        qc.update_quantum_state(qs)
        qc = None
        qs = None
        for gate in gates:
            gate = None
        for pgate in parametric_gates:
            gate = None

        gates = None
        parametric_gates = None

    def test_add_same_gate_multiple_time(self):
        from qulacs import QuantumCircuit, QuantumState
        from qulacs.gate import X, DepolarizingNoise, DephasingNoise, Probabilistic, RX
        state = QuantumState(1)
        circuit = QuantumCircuit(1)
        noise = DepolarizingNoise(0, 0)
        circuit.add_gate(noise)
        circuit.add_gate(noise.copy())
        circuit.add_gate(DephasingNoise(0, 0))
        circuit.add_gate(Probabilistic([0.1], [RX(0, 0)]))
        gate = RX(0, 0)
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
        del term

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
        matrix = lil_matrix((4, 4), dtype=np.complex128)
        matrix[0, 0] = 1 + 1.j
        matrix[1, 1] = 1. + 1.j
        gate = SparseMatrix([0, 1], matrix)
        gate.update_quantum_state(state)
        del gate
        del state

    def test_copied_parametric_gate(self):

        from qulacs import ParametricQuantumCircuit, QuantumState
        from qulacs.gate import ParametricRX

        def f():
            circuit = ParametricQuantumCircuit(1)
            gate = ParametricRX(0, 0.1)
            circuit.add_parametric_gate(gate)
            circuit.add_parametric_gate(gate)
            circuit.add_gate(gate)
            gate.set_parameter_value(0.2)
            circuit.add_parametric_gate(gate)
            circuit.add_parametric_RX_gate(0, 0.3)
            gate2 = gate.copy()
            gate2.set_parameter_value(0.4)
            gate.set_parameter_value(1.0)
            del gate
            circuit.add_parametric_gate(gate2)
            circuit.remove_gate(1)
            del gate2
            return circuit

        c = f()
        for gc in range(c.get_parameter_count()):
            val = c.get_parameter(gc)
            c.set_parameter(gc, val + 1.0)
            self.assertAlmostEqual(val, gc * 0.1 + 0.1, msg="check vector size")

        d = c.copy()
        del c
        for gc in range(d.get_parameter_count()):
            val = d.get_parameter(gc)
            d.set_parameter(gc, val + 10)
            val = d.get_parameter(gc)
            self.assertAlmostEqual(val, 11.1 + gc * 0.1, msg="check vector size")

        qs = QuantumState(1)
        d.update_quantum_state(qs)
        del d
        del qs

    def test_parametric_gate_position(self):

        from qulacs import ParametricQuantumCircuit, QuantumState
        from qulacs.gate import ParametricRX

        def check(pqc, idlist):
            cnt = pqc.get_parameter_count()
            self.assertEqual(cnt, len(idlist))
            for ind in range(cnt):
                pos = pqc.get_parametric_gate_position(ind)
                self.assertEqual(pos, idlist[ind])

        pqc = ParametricQuantumCircuit(1)
        gate = ParametricRX(0, 0.1)
        pqc.add_parametric_gate(gate)  # [0]
        check(pqc, [0])
        pqc.add_parametric_gate(gate)  # [0, 1]
        check(pqc, [0, 1])
        pqc.add_gate(gate)  # [0, 1, *]
        check(pqc, [0, 1])
        pqc.add_parametric_gate(gate, 0)  # [2, 0, 1, *]
        check(pqc, [1, 2, 0])
        pqc.add_gate(gate, 0)  # [*, 2, 0, 1, *]
        check(pqc, [2, 3, 1])
        pqc.add_parametric_gate(gate, 0)  # [3, *, 2, 0, 1, *]
        check(pqc, [3, 4, 2, 0])
        pqc.remove_gate(4)  # [2, *, 1, 0, *]
        check(pqc, [3, 2, 0])
        pqc.remove_gate(1)  # [2, 1, 0, *]
        check(pqc, [2, 1, 0])
        pqc.add_parametric_gate(gate)  # [2, 1, 0, *, 3]
        check(pqc, [2, 1, 0, 4])
        pqc.add_parametric_gate(gate, 2)  # [2, 1, 4, 0, *, 3]
        check(pqc, [3, 1, 0, 5, 2])
        pqc.remove_gate(3)  # [1, 0, 3, *, 2]
        check(pqc, [1, 0, 4, 2])


if __name__ == "__main__":
    unittest.main()
