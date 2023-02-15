
# set library dir
import sys
import unittest
import warnings

import numpy as np

import qulacs

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
        self.assertTrue(((vector - vector_ans) < 1e-10).all(),
                        msg="check set_zero_state")

    def test_comp_basis(self):
        pos = 0b0101
        self.state.set_computational_basis(pos)
        vector = self.state.get_vector()
        vector_ans = np.zeros(self.dim)
        vector_ans[pos] = 1.
        self.assertTrue(((vector - vector_ans) < 1e-10).all(),
                        msg="check set_computational_basis")


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
        self.assertTrue(((vector - vector_ans) < 1e-10).all(),
                        msg="check make bell state")


class TestObservable(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_matrix(self):
        import numpy as np

        from qulacs import Observable
        n_qubits = 3
        obs = Observable(n_qubits)
        obs.add_operator(.5, "Z 2")
        obs.add_operator(1., "X 0 X 1 X 2")
        obs.add_operator(1., "Y 1")
        ans = np.array(
            [
                [.5, 0, -1j, 0, 0, 0, 0, 1],
                [0, .5, 0, -1j, 0, 0, 1, 0],
                [1j, 0, .5, 0, 0, 1, 0, 0],
                [0, 1j, 0, .5, 1, 0, 0, 0],
                [0, 0, 0, 1, -.5, 0, -1j, 0],
                [0, 0, 1, 0, 0, -.5, 0, -1j],
                [0, 1, 0, 0, 1j, 0, -.5, 0],
                [1, 0, 0, 0, 0, 1j, 0, -.5],
            ], dtype=np.complex128
        )
        self.assertLessEqual(np.linalg.norm(
            ans-obs.get_matrix().todense()), 1e-6)
        from qulacs import GeneralQuantumOperator
        obs = GeneralQuantumOperator(n_qubits)
        obs.add_operator(.5j, "Z 2")
        obs.add_operator(1., "X 0 X 1 X 2")
        obs.add_operator(1., "Y 1")
        ans = np.array(
            [
                [.5j, 0, -1j, 0, 0, 0, 0, 1],
                [0, .5j, 0, -1j, 0, 0, 1, 0],
                [1j, 0, .5j, 0, 0, 1, 0, 0],
                [0, 1j, 0, .5j, 1, 0, 0, 0],
                [0, 0, 0, 1, -.5j, 0, -1j, 0],
                [0, 0, 1, 0, 0, -.5j, 0, -1j],
                [0, 1, 0, 0, 1j, 0, -.5j, 0],
                [1, 0, 0, 0, 0, 1j, 0, -.5j],
            ], dtype=np.complex128
        )
        self.assertLessEqual(np.linalg.norm(
            ans-obs.get_matrix().todense()), 1e-6)


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
        from scipy.sparse import lil_matrix

        from qulacs import QuantumCircuit, QuantumState
        from qulacs.gate import (CNOT, CPTP, CZ, FREDKIN, P0, P1, RX, RY, RZ,
                                 SWAP, TOFFOLI, U1, U2, U3, Adaptive,
                                 AmplitudeDampingNoise, BitFlipNoise,
                                 DenseMatrix, DephasingNoise,
                                 DepolarizingNoise, DiagonalMatrix, H,
                                 Identity, IndependentXZNoise, Instrument,
                                 Measurement, Pauli, PauliRotation,
                                 Probabilistic, RandomUnitary,
                                 ReversibleBoolean, S, Sdag, SparseMatrix,
                                 StateReflection, T, Tdag,
                                 TwoQubitDepolarizingNoise, X, Y, Z, add,
                                 merge, sqrtX, sqrtXdag, sqrtY, sqrtYdag,
                                 to_matrix_gate)
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
            Identity(0), X(0), Y(0), Z(0), H(0), S(0), Sdag(0), T(
                0), Tdag(0), sqrtX(0), sqrtXdag(0), sqrtY(0), sqrtYdag(0),
            Probabilistic([0.5, 0.5], [X(0), Y(0)]), CPTP(
                [P0(0), P1(0)]), Instrument([P0(0), P1(0)], 1), Adaptive(X(0), adap),
            CNOT(0, 1), CZ(0, 1), SWAP(0, 1), TOFFOLI(0, 1, 2), FREDKIN(
                0, 1, 2), Pauli([0, 1], [1, 2]), PauliRotation([0, 1], [1, 2], 0.1),
            DenseMatrix(0, np.eye(2)), DenseMatrix(
                [0, 1], np.eye(4)), SparseMatrix([0, 1], sparse_mat),
            DiagonalMatrix([0, 1], np.ones(4)), RandomUnitary(
                [0, 1]), ReversibleBoolean([0, 1], func), StateReflection(ref),
            BitFlipNoise(0, 0.1), DephasingNoise(0, 0.1), IndependentXZNoise(
                0, 0.1), DepolarizingNoise(0, 0.1), TwoQubitDepolarizingNoise(0, 1, 0.1),
            AmplitudeDampingNoise(0, 0.1), Measurement(0, 1), merge(
                X(0), Y(1)), add(X(0), Y(1)), to_matrix_gate(X(0)),
            P0(0), P1(0), U1(0, 0.), U2(0, 0., 0.), U3(
                0, 0., 0., 0.), RX(0, 0.), RY(0, 0.), RZ(0, 0.),
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
        from scipy.sparse import lil_matrix

        from qulacs import ParametricQuantumCircuit, QuantumState
        from qulacs.gate import (CNOT, CPTP, CZ, FREDKIN, P0, P1, RX, RY, RZ,
                                 SWAP, TOFFOLI, U1, U2, U3, Adaptive,
                                 AmplitudeDampingNoise, BitFlipNoise,
                                 DenseMatrix, DephasingNoise,
                                 DepolarizingNoise, DiagonalMatrix, H,
                                 Identity, IndependentXZNoise, Instrument,
                                 Measurement, ParametricPauliRotation,
                                 ParametricRX, ParametricRY, ParametricRZ,
                                 Pauli, PauliRotation, Probabilistic,
                                 RandomUnitary, ReversibleBoolean, S, Sdag,
                                 SparseMatrix, StateReflection, T, Tdag,
                                 TwoQubitDepolarizingNoise, X, Y, Z, add,
                                 merge, sqrtX, sqrtXdag, sqrtY, sqrtYdag,
                                 to_matrix_gate)
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
            Identity(0), X(0), Y(0), Z(0), H(0), S(0), Sdag(0), T(
                0), Tdag(0), sqrtX(0), sqrtXdag(0), sqrtY(0), sqrtYdag(0),
            Probabilistic([0.5, 0.5], [X(0), Y(0)]), CPTP(
                [P0(0), P1(0)]), Instrument([P0(0), P1(0)], 1), Adaptive(X(0), adap),
            CNOT(0, 1), CZ(0, 1), SWAP(0, 1), TOFFOLI(0, 1, 2), FREDKIN(
                0, 1, 2), Pauli([0, 1], [1, 2]), PauliRotation([0, 1], [1, 2], 0.1),
            DenseMatrix(0, np.eye(2)), DenseMatrix(
                [0, 1], np.eye(4)), SparseMatrix([0, 1], sparse_mat),
            DiagonalMatrix([0, 1], np.ones(4)), RandomUnitary(
                [0, 1]), ReversibleBoolean([0, 1], func), StateReflection(ref),
            BitFlipNoise(0, 0.1), DephasingNoise(0, 0.1), IndependentXZNoise(
                0, 0.1), DepolarizingNoise(0, 0.1), TwoQubitDepolarizingNoise(0, 1, 0.1),
            AmplitudeDampingNoise(0, 0.1), Measurement(0, 1), merge(
                X(0), Y(1)), add(X(0), Y(1)), to_matrix_gate(X(0)),
            P0(0), P1(0), U1(0, 0.), U2(0, 0., 0.), U3(
                0, 0., 0., 0.), RX(0, 0.), RY(0, 0.), RZ(0, 0.),
        ]

        gates.append(merge(gates[0], gates[1]))
        gates.append(add(gates[0], gates[1]))

        parametric_gates = [
            ParametricRX(0, 0.1), ParametricRY(0, 0.1), ParametricRZ(
                0, 0.1), ParametricPauliRotation([0, 1], [1, 1], 0.1)
        ]

        ref = None
        for gate in gates:
            qc.add_gate(gate)
            gate.update_quantum_state(qs)

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
        from qulacs.gate import (RX, DephasingNoise, DepolarizingNoise,
                                 Probabilistic, X)
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

        from scipy.sparse import lil_matrix

        from qulacs import QuantumState
        from qulacs.gate import SparseMatrix
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
            self.assertAlmostEqual(val, gc * 0.1 + 0.1,
                                   msg="check vector size")

        d = c.copy()
        del c
        for gc in range(d.get_parameter_count()):
            val = d.get_parameter(gc)
            d.set_parameter(gc, val + 10)
            val = d.get_parameter(gc)
            self.assertAlmostEqual(val, 11.1 + gc * 0.1,
                                   msg="check vector size")

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


class TestDensityMatrixHandling(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_density_matrix(self):
        num_qubit = 5
        sv = qulacs.StateVector(num_qubit)
        dm = qulacs.DensityMatrix(num_qubit)
        sv.set_Haar_random_state(seed=0)
        dm.load(sv)
        svv = np.atleast_2d(sv.get_vector()).T
        mat = np.dot(svv, svv.T.conj())
        self.assertTrue(np.allclose(dm.get_matrix(), mat),
                        msg="check pure matrix to density matrix")

    def test_tensor_product_sv(self):
        num_qubit = 4
        sv1 = qulacs.StateVector(num_qubit)
        sv2 = qulacs.StateVector(num_qubit)
        sv1.set_Haar_random_state(seed=0)
        sv2.set_Haar_random_state(seed=1)
        sv3 = qulacs.state.tensor_product(sv1, sv2)
        sv3_test = np.kron(sv1.get_vector(), sv2.get_vector())
        self.assertTrue(np.allclose(sv3_test, sv3.get_vector()),
                        msg="check pure state tensor product")
        del sv1
        del sv2
        del sv3

    def test_tensor_product_dm(self):
        num_qubit = 4
        dm1 = qulacs.DensityMatrix(num_qubit)
        dm2 = qulacs.DensityMatrix(num_qubit)
        dm1.set_Haar_random_state(seed=0)
        dm2.set_Haar_random_state(seed=1)
        dm3 = qulacs.state.tensor_product(dm1, dm2)
        dm3_test = np.kron(dm1.get_matrix(), dm2.get_matrix())
        self.assertTrue(np.allclose(dm3_test, dm3.get_matrix()),
                        msg="check density matrix tensor product")
        del dm1
        del dm2
        del dm3

    def test_tensor_product_different_size_sv(self):
        num_qubit = 4
        sv1 = qulacs.StateVector(num_qubit)
        sv2 = qulacs.StateVector(num_qubit + 1)
        sv1.set_Haar_random_state(seed=0)
        sv2.set_Haar_random_state(seed=1)
        sv3 = qulacs.state.tensor_product(sv1, sv2)
        sv3_test = np.kron(sv1.get_vector(), sv2.get_vector())
        self.assertTrue(np.allclose(sv3_test, sv3.get_vector()),
                        msg="check pure state tensor product")
        del sv1
        del sv2
        del sv3

    def test_tensor_product_different_size_dm(self):
        num_qubit = 4
        dm1 = qulacs.DensityMatrix(num_qubit)
        dm2 = qulacs.DensityMatrix(num_qubit + 1)
        dm1.set_Haar_random_state(seed=0)
        dm2.set_Haar_random_state(seed=1)
        dm3 = qulacs.state.tensor_product(dm1, dm2)
        dm3_test = np.kron(dm1.get_matrix(), dm2.get_matrix())
        self.assertTrue(np.allclose(dm3_test, dm3.get_matrix()),
                        msg="check density matrix tensor product")
        del dm1
        del dm2
        del dm3

    def test_permutate_qubit_sv(self):
        num_qubit = 8
        sv = qulacs.StateVector(num_qubit)
        sv.set_Haar_random_state(seed=0)
        order = np.arange(num_qubit)
        np.random.shuffle(order)

        arr = []
        for ind in range(2**num_qubit):
            s = format(ind, "0{}b".format(num_qubit))
            s = np.array(list(s[::-1]))
            v = np.array(["*"]*num_qubit)
            for ind in range(len(s)):
                v[order[ind]] = s[ind]
            s = ("".join(v))[::-1]
            arr.append(int(s, 2))

        sv_perm = qulacs.state.permutate_qubit(sv, order)
        self.assertTrue(np.allclose(sv.get_vector()[
                        arr], sv_perm.get_vector()), msg="check pure state permutation")
        del sv_perm
        del sv

    def test_permutate_qubit_dm(self):
        num_qubit = 3
        dm = qulacs.DensityMatrix(num_qubit)
        dm.set_Haar_random_state(seed=0)
        order = np.arange(num_qubit)
        np.random.shuffle(order)

        arr = []
        for ind in range(2**num_qubit):
            s = format(ind, "0{}b".format(num_qubit))
            s = np.array(list(s[::-1]))
            v = np.array(["*"]*num_qubit)
            for ind in range(len(s)):
                v[order[ind]] = s[ind]
            s = ("".join(v))[::-1]
            arr.append(int(s, 2))

        dm_perm = qulacs.state.permutate_qubit(dm, order)
        dm_perm_test = dm.get_matrix()
        dm_perm_test = dm_perm_test[arr, :]
        dm_perm_test = dm_perm_test[:, arr]
        self.assertTrue(np.allclose(dm_perm_test, dm_perm.get_matrix()),
                        msg="check density matrix permutation")
        del dm_perm
        del dm

    def test_partial_trace_dm(self):
        num_qubit = 5
        num_traceout = 2
        dm = qulacs.DensityMatrix(num_qubit)
        dm.set_Haar_random_state(seed=0)
        mat = dm.get_matrix()

        target = np.arange(num_qubit)
        np.random.shuffle(target)
        target = target[:num_traceout]
        target_cor = [num_qubit-1-i for i in target]
        target_cor.sort()

        dmt = mat.reshape([2, 2]*num_qubit)
        for cnt, val in enumerate(target_cor):
            ofs = num_qubit - cnt
            dmt = np.trace(dmt, axis1=val-cnt, axis2=ofs+val-cnt)
        dmt = dmt.reshape([2**(num_qubit-num_traceout),
                          2**(num_qubit-num_traceout)])

        pdm = qulacs.state.partial_trace(dm, target)
        self.assertTrue(np.allclose(pdm.get_matrix(), dmt),
                        msg="check density matrix partial trace")
        del dm, pdm

    def test_partial_trace_sv(self):
        num_qubit = 6
        num_traceout = 4
        sv = qulacs.StateVector(num_qubit)
        sv.set_Haar_random_state(seed=0)
        svv = np.atleast_2d(sv.get_vector()).T
        mat = np.dot(svv, svv.T.conj())

        target = np.arange(num_qubit)
        np.random.shuffle(target)
        target = target[:num_traceout]
        target_cor = [num_qubit-1-i for i in target]
        target_cor.sort()

        dmt = mat.reshape([2, 2]*num_qubit)
        for cnt, val in enumerate(target_cor):
            ofs = num_qubit - cnt
            dmt = np.trace(dmt, axis1=val-cnt, axis2=ofs+val-cnt)
        dmt = dmt.reshape([2**(num_qubit-num_traceout),
                          2**(num_qubit-num_traceout)])

        pdm = qulacs.state.partial_trace(sv, target)
        self.assertTrue(np.allclose(pdm.get_matrix(), dmt),
                        msg="check pure state partial trace")


class TestUtils(unittest.TestCase):
    def setUp(self):
        # suppress warning from openfermion/cirq
        warnings.simplefilter('ignore', category=DeprecationWarning)

    def tearDown(self):
        pass

    def test_convert_openfermion_op(self):
        from openfermion import QubitOperator

        from qulacs.utils import convert_openfermion_op
        openfermion_op = QubitOperator()
        openfermion_op += 1. * QubitOperator("X0")
        openfermion_op += 2. * QubitOperator("Z0 Y1")
        openfermion_op += 1. * QubitOperator("Z0 Y3")

        qulacs_op = convert_openfermion_op(openfermion_op)
        str_qulacs_op = str(qulacs_op)
        # operator ordering in openfermion may differ sometimes so we have to do this
        str_qulacs_op = str_qulacs_op.replace("(1,0) X 0", "")
        str_qulacs_op = str_qulacs_op.replace("(2,0) Z 0 Y 1", "")
        str_qulacs_op = str_qulacs_op.replace("(1,0) Z 0 Y 3", "")
        self.assertEqual(str_qulacs_op, " +  + ")

    def test_GQO(self):
        self.n = 4
        self.dim = 2**self.n
        stateA = qulacs.QuantumState(self.n)
        stateB = qulacs.QuantumState(self.n)
        stateC = qulacs.QuantumState(self.n)
        stateB.set_Haar_random_state()
        RanGate = qulacs.gate.RandomUnitary([0, 1, 2, 3])
        GQO_ret = qulacs.to_general_quantum_operator(RanGate, 4, 0.00001)
        GQO_ret.apply_to_state(stateA, stateB, stateC)
        RanGate.update_quantum_state(stateB)

        inpro = qulacs.state.inner_product(stateB, stateC).real
        self.assertTrue(inpro > 0.99, msg="GQO_test")


class TestDensityMatrixHandling(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_density_matrix(self):
        num_qubit = 5
        sv = qulacs.StateVector(num_qubit)
        dm = qulacs.DensityMatrix(num_qubit)
        sv.set_Haar_random_state(seed=0)
        dm.load(sv)
        svv = np.atleast_2d(sv.get_vector()).T
        mat = np.dot(svv, svv.T.conj())
        self.assertTrue(np.allclose(dm.get_matrix(), mat),
                        msg="check pure matrix to density matrix")

    def test_tensor_product_sv(self):
        num_qubit = 4
        sv1 = qulacs.StateVector(num_qubit)
        sv2 = qulacs.StateVector(num_qubit)
        sv1.set_Haar_random_state(seed=0)
        sv2.set_Haar_random_state(seed=1)
        sv3 = qulacs.state.tensor_product(sv1, sv2)
        sv3_test = np.kron(sv1.get_vector(), sv2.get_vector())
        self.assertTrue(np.allclose(sv3_test, sv3.get_vector()),
                        msg="check pure state tensor product")
        del sv1
        del sv2
        del sv3

    def test_tensor_product_dm(self):
        num_qubit = 4
        dm1 = qulacs.DensityMatrix(num_qubit)
        dm2 = qulacs.DensityMatrix(num_qubit)
        dm1.set_Haar_random_state(seed=0)
        dm2.set_Haar_random_state(seed=1)
        dm3 = qulacs.state.tensor_product(dm1, dm2)
        dm3_test = np.kron(dm1.get_matrix(), dm2.get_matrix())
        self.assertTrue(np.allclose(dm3_test, dm3.get_matrix()),
                        msg="check density matrix tensor product")
        del dm1
        del dm2
        del dm3

    def test_tensor_product_different_size_sv(self):
        num_qubit = 4
        sv1 = qulacs.StateVector(num_qubit)
        sv2 = qulacs.StateVector(num_qubit + 1)
        sv1.set_Haar_random_state(seed=0)
        sv2.set_Haar_random_state(seed=1)
        sv3 = qulacs.state.tensor_product(sv1, sv2)
        sv3_test = np.kron(sv1.get_vector(), sv2.get_vector())
        self.assertTrue(np.allclose(sv3_test, sv3.get_vector()),
                        msg="check pure state tensor product")
        del sv1
        del sv2
        del sv3

    def test_tensor_product_different_size_dm(self):
        num_qubit = 4
        dm1 = qulacs.DensityMatrix(num_qubit)
        dm2 = qulacs.DensityMatrix(num_qubit + 1)
        dm1.set_Haar_random_state(seed=0)
        dm2.set_Haar_random_state(seed=1)
        dm3 = qulacs.state.tensor_product(dm1, dm2)
        dm3_test = np.kron(dm1.get_matrix(), dm2.get_matrix())
        self.assertTrue(np.allclose(dm3_test, dm3.get_matrix()),
                        msg="check density matrix tensor product")
        del dm1
        del dm2
        del dm3

    def test_permutate_qubit_sv(self):
        num_qubit = 8
        sv = qulacs.StateVector(num_qubit)
        sv.set_Haar_random_state(seed=0)
        order = np.arange(num_qubit)
        np.random.shuffle(order)

        arr = []
        for ind in range(2**num_qubit):
            s = format(ind, "0{}b".format(num_qubit))
            s = np.array(list(s[::-1]))
            v = np.array(["*"]*num_qubit)
            for ind in range(len(s)):
                v[order[ind]] = s[ind]
            s = ("".join(v))[::-1]
            arr.append(int(s, 2))

        sv_perm = qulacs.state.permutate_qubit(sv, order)
        self.assertTrue(np.allclose(sv.get_vector()[
                        arr], sv_perm.get_vector()), msg="check pure state permutation")
        del sv_perm
        del sv

    def test_permutate_qubit_dm(self):
        num_qubit = 3
        dm = qulacs.DensityMatrix(num_qubit)
        dm.set_Haar_random_state(seed=0)
        order = np.arange(num_qubit)
        np.random.shuffle(order)

        arr = []
        for ind in range(2**num_qubit):
            s = format(ind, "0{}b".format(num_qubit))
            s = np.array(list(s[::-1]))
            v = np.array(["*"]*num_qubit)
            for ind in range(len(s)):
                v[order[ind]] = s[ind]
            s = ("".join(v))[::-1]
            arr.append(int(s, 2))

        dm_perm = qulacs.state.permutate_qubit(dm, order)
        dm_perm_test = dm.get_matrix()
        dm_perm_test = dm_perm_test[arr, :]
        dm_perm_test = dm_perm_test[:, arr]
        self.assertTrue(np.allclose(dm_perm_test, dm_perm.get_matrix()),
                        msg="check density matrix permutation")
        del dm_perm
        del dm

    def test_partial_trace_dm(self):
        num_qubit = 5
        num_traceout = 2
        dm = qulacs.DensityMatrix(num_qubit)
        dm.set_Haar_random_state(seed=0)
        mat = dm.get_matrix()

        target = np.arange(num_qubit)
        np.random.shuffle(target)
        target = target[:num_traceout]
        target_cor = [num_qubit-1-i for i in target]
        target_cor.sort()

        dmt = mat.reshape([2, 2]*num_qubit)
        for cnt, val in enumerate(target_cor):
            ofs = num_qubit - cnt
            dmt = np.trace(dmt, axis1=val-cnt, axis2=ofs+val-cnt)
        dmt = dmt.reshape([2**(num_qubit-num_traceout),
                          2**(num_qubit-num_traceout)])

        pdm = qulacs.state.partial_trace(dm, target)
        self.assertTrue(np.allclose(pdm.get_matrix(), dmt),
                        msg="check density matrix partial trace")
        del dm, pdm

    def test_partial_trace_sv(self):
        num_qubit = 6
        num_traceout = 4
        sv = qulacs.StateVector(num_qubit)
        sv.set_Haar_random_state(seed=0)
        svv = np.atleast_2d(sv.get_vector()).T
        mat = np.dot(svv, svv.T.conj())

        target = np.arange(num_qubit)
        np.random.shuffle(target)
        target = target[:num_traceout]
        target_cor = [num_qubit-1-i for i in target]
        target_cor.sort()

        dmt = mat.reshape([2, 2]*num_qubit)
        for cnt, val in enumerate(target_cor):
            ofs = num_qubit - cnt
            dmt = np.trace(dmt, axis1=val-cnt, axis2=ofs+val-cnt)
        dmt = dmt.reshape([2**(num_qubit-num_traceout),
                          2**(num_qubit-num_traceout)])

        pdm = qulacs.state.partial_trace(sv, target)
        self.assertTrue(np.allclose(pdm.get_matrix(), dmt),
                        msg="check pure state partial trace")


class TestNoiseSimulator(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_noise_simulator(self):
        def get_heavy_output_probability(n, depth, error_prob, shots=1000):
            one_qubit_noise = ["Depolarizing", "BitFlip",
                               "Dephasing", "IndependentXZ", "AmplitudeDamping"]
            two_qubit_noise = ["Depolarizing"]
            from qulacs import NoiseSimulator, QuantumState
            from qulacs.gate import CNOT, CZ, T, sqrtX, sqrtY
            circuit_with_noise = qulacs.QuantumCircuit(n)
            circuit_without_noise = qulacs.QuantumCircuit(n)
            for d in range(depth):
                for i in range(n):
                    r = np.random.randint(0, 5)
                    if r == 0:
                        circuit_with_noise.add_noise_gate(
                            sqrtX(i), one_qubit_noise[np.random.randint(0, 5)], error_prob)
                        circuit_without_noise.add_sqrtX_gate(i)
                    elif r == 1:
                        circuit_with_noise.add_noise_gate(
                            sqrtY(i), one_qubit_noise[np.random.randint(0, 5)], error_prob)
                        circuit_without_noise.add_sqrtY_gate(i)
                    elif r == 2:
                        circuit_with_noise.add_noise_gate(
                            T(i), one_qubit_noise[np.random.randint(0, 5)], error_prob)
                        circuit_without_noise.add_T_gate(i)
                    elif r == 3:
                        if i + 1 < n:
                            circuit_with_noise.add_noise_gate(
                                CNOT(i, i + 1), two_qubit_noise[np.random.randint(0, 1)], error_prob)
                            circuit_without_noise.add_CNOT_gate(i, i + 1)
                    elif r == 4:
                        if i + 1 < n:
                            circuit_with_noise.add_noise_gate(
                                CZ(i, i + 1), two_qubit_noise[np.random.randint(0, 1)], error_prob)
                            circuit_without_noise.add_CZ_gate(i, i + 1)

            ideal_state = qulacs.QuantumState(n)
            circuit_without_noise.update_quantum_state(ideal_state)
            prob_dist = [abs(x)**2 for x in ideal_state.get_vector()]
            p_median = np.sort(prob_dist)[2**(n-1) - 1]
            heavy_output = set()
            for i in range(2**n):
                if prob_dist[i] > p_median:
                    heavy_output.add(i)

            sim = NoiseSimulator(circuit_with_noise, QuantumState(n))
            noisy_sample = sim.execute(shots)
            num_heavy_output = 0
            for sample in noisy_sample:
                if sample in heavy_output:
                    num_heavy_output += 1
            return num_heavy_output / shots, heavy_output, noisy_sample

        low_noise_prob, low_noise_heavy_output, low_noise_result = get_heavy_output_probability(
            7, 100, 1e-5)
        high_noise_prob, high_noise_heavy_output, high_noise_result = get_heavy_output_probability(
            7, 100, 0.01)
        if low_noise_prob < 2/3:
            print(
                f"[ERROR] On low noise environment Heavy Output percentage should be > 0.666, but was {low_noise_prob}")
            print("Telemetry Information:")
            print(f"Sampling Result: {low_noise_result}")
            print(f"Heavy Output: {low_noise_heavy_output}")
        if high_noise_prob > 2/3:
            print(
                f"[ERROR] On high noise environment Heavy Output percentage should be < 0.666, but was {high_noise_prob}")
            print("Telemetry Information:")
            print(f"Sampling Result: {high_noise_result}")
            print(f"Heavy Output: {high_noise_heavy_output}")

        self.assertGreater(low_noise_prob, 2 / 3)
        self.assertLess(high_noise_prob, 2 / 3)


class TestQASM(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_qasm_converter(self):
        import numpy as np

        from qulacs import QuantumCircuit
        from qulacs.converter import (convert_QASM_to_qulacs_circuit,
                                      convert_qulacs_circuit_to_QASM)
        from qulacs.gate import (CNOT, CZ, FREDKIN, RX, RY, RZ, SWAP, TOFFOLI,
                                 U1, U2, U3, DenseMatrix, H, Identity,
                                 RandomUnitary, S, Sdag, T, Tdag, X, Y, Z, add,
                                 merge, sqrtX, sqrtXdag, to_matrix_gate)
        gates = [
            Identity(0), X(0), Y(0), Z(0), H(0), S(0), Sdag(0), T(
                0), Tdag(0), sqrtX(0), sqrtXdag(0),
            CNOT(0, 1), CZ(0, 1), SWAP(0, 1), TOFFOLI(0, 1, 2), FREDKIN(
                0, 1, 2),
            DenseMatrix(0, np.eye(2)), DenseMatrix(
                [0, 1], np.eye(4)), RandomUnitary(
                [0, 1]), merge(
                X(0), Y(1)), add(X(0), Y(1)), to_matrix_gate(X(0)),
            U1(0, 0.), U2(0, 0., 0.), U3(
                0, 0., 0., 0.), RX(0, 0.), RY(0, 0.), RZ(0, 0.),
        ]
        circuit = QuantumCircuit(5)
        for x in gates:
            circuit.add_gate(x)
        QASM_strs = convert_qulacs_circuit_to_QASM(circuit)
        transpiled_circuit = convert_QASM_to_qulacs_circuit(QASM_strs)
        for x in range(transpiled_circuit.get_gate_count()):
            assert np.allclose(transpiled_circuit.get_gate(x).get_matrix(),
                               gates[x].get_matrix())


class TestJSON(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_operator(self):
        import json
        import random

        from qulacs import (GeneralQuantumOperator, Observable, PauliOperator,
                            QuantumState, observable, quantum_operator)

        n = 5

        def random_pauli_operator():
            op = PauliOperator((random.random()*2-1) +
                               (random.random()*2-1) * 1j)
            for _ in range(random.randint(1, 5)):
                op.add_single_Pauli(
                    random.randint(0, n-1), random.randint(0, 3)
                )
            return op

        def random_hermitian_pauli_operator():
            op = PauliOperator(random.random()*2-1)
            for _ in range(random.randint(1, 5)):
                op.add_single_Pauli(
                    random.randint(0, n-1), random.randint(0, 3)
                )
            return op

        oridinal_operator = GeneralQuantumOperator(5)
        for _ in range(5):
            oridinal_operator.add_operator(random_pauli_operator())

        json_string = oridinal_operator.to_json()
        json.loads(json_string)
        restored_operator = quantum_operator.from_json(json_string)

        for _ in range(3):
            state = QuantumState(n)
            self.assertAlmostEqual(
                oridinal_operator.get_expectation_value(state),
                restored_operator.get_expectation_value(state)
            )

        oridinal_observable = Observable(n)
        for _ in range(5):
            oridinal_observable.add_operator(random_hermitian_pauli_operator())

        json_string = oridinal_observable.to_json()
        json.loads(json_string)
        restored_observable = observable.from_json(json_string)

        for _ in range(3):
            state = QuantumState(n)
            state.set_Haar_random_state()
            self.assertAlmostEqual(
                oridinal_observable.get_expectation_value(state),
                restored_observable.get_expectation_value(state)
            )

        non_hermitian_operator = GeneralQuantumOperator(1)
        non_hermitian_operator.add_operator(1j, "X 0")
        with self.assertRaises(RuntimeError):
            observable.from_json(non_hermitian_operator.to_json())

    def test_gate(self):
        import json
        import random

        from scipy.sparse import lil_matrix

        from qulacs import QuantumState, gate
        from qulacs.gate import (CNOT, CZ, FREDKIN, P0, P1, RX, RY, RZ, SWAP,
                                 TOFFOLI, U1, U2, U3, H, Identity, Pauli,
                                 PauliRotation, S, Sdag, StateReflection, T,
                                 Tdag, X, Y, Z, add, merge, sqrtX, sqrtXdag,
                                 sqrtY, sqrtYdag, to_matrix_gate)

        n = 3

        def execute_test_gate():
            qs = QuantumState(n)
            sparse_mat = lil_matrix((4, 4))
            sparse_mat[0, 0] = 1
            sparse_mat[1, 1] = 1

            axis = QuantumState(n)
            axis.set_Haar_random_state()

            gates = [
                Identity(0), X(0), Y(0), Z(0), H(0), S(0), Sdag(0), T(
                    0), Tdag(0), sqrtX(0), sqrtXdag(0),
                sqrtY(0), sqrtYdag(0),
                CNOT(0, 1), CZ(0, 1), SWAP(0, 1), TOFFOLI(0, 1, 2), FREDKIN(
                    0, 1, 2), Pauli([0, 1], [1, 2]), PauliRotation([0, 1], [1, 2], random.random()),
                StateReflection(axis),
                merge(
                    X(0), Y(1)), add(X(0), Y(1)), to_matrix_gate(X(0)),
                P0(0), P1(0), U1(0, random.random()), U2(0, random.random(), random.random()), U3(
                    0, random.random(), random.random(), random.random()), RX(0, random.random()), RY(0, random.random()), RZ(0, random.random()),
            ]
            gates.append(merge(Identity(0), X(0)))
            gates.append(add(Identity(0), X(0)))

            for g in gates:
                qs.set_Haar_random_state()
                qs_json = qs.copy()
                g.update_quantum_state(qs)
                json_string = g.to_json()
                json.loads(json_string)
                g_json = gate.from_json(json_string)
                g_json.update_quantum_state(qs_json)
                for i in range(n):
                    self.assertAlmostEqual(qs.get_zero_probability(
                        i), qs_json.get_zero_probability(i))

        for _ in range(10):
            execute_test_gate()

    def test_parametric_gate(self):
        import json
        import random

        from qulacs import QuantumState, gate
        from qulacs.gate import (ParametricPauliRotation, ParametricRX,
                                 ParametricRY, ParametricRZ)

        n = 3
        qs = QuantumState(n)

        gates = [
            ParametricRX(0, random.random()), ParametricRY(0, random.random()), ParametricRZ(
                0, random.random()), ParametricPauliRotation([0, 1], [1, 1], random.random())
        ]

        for g in gates:
            qs.set_Haar_random_state()
            qs_json = qs.copy()
            g.update_quantum_state(qs)
            json_string = g.to_json()
            json.loads(json_string)
            g_json = gate.from_json(json_string)
            g_json.update_quantum_state(qs_json)
            for i in range(n):
                self.assertAlmostEqual(qs.get_zero_probability(
                    i), qs_json.get_zero_probability(i))

    def test_matrix_gate(self):
        import json
        import random

        from scipy.sparse import csc_matrix

        from qulacs import QuantumState, gate
        from qulacs.gate import DenseMatrix, SparseMatrix

        n = 3

        def execute_test_matrix_gate():
            qs = QuantumState(n)

            # DenseMatrix
            matrix = np.array(
                [[random.random(), random.random()], [random.random(), random.random()]])
            g = DenseMatrix(0, matrix)
            qs.set_Haar_random_state()
            qs_json = qs.copy()
            g.update_quantum_state(qs)
            json_string = g.to_json()
            json.loads(json_string)
            g_json = gate.from_json(json_string)
            g_json.update_quantum_state(qs_json)
            for i in range(n):
                self.assertAlmostEqual(qs.get_zero_probability(
                    i), qs_json.get_zero_probability(i))

            # SparseMatrix
            matrix = np.array(
                [[random.random(), random.random()], [random.random(), random.random()]])
            csc = csc_matrix(matrix)
            g = SparseMatrix([0], csc)
            qs.set_Haar_random_state()
            qs_json = qs.copy()
            g.update_quantum_state(qs)
            json_string = g.to_json()
            json.loads(json_string)
            g_json = gate.from_json(json_string)
            g_json.update_quantum_state(qs_json)
            for i in range(n):
                self.assertAlmostEqual(qs.get_zero_probability(
                    i), qs_json.get_zero_probability(i))

        for _ in range(10):
            execute_test_matrix_gate()

    def test_probabilistic_gate(self):
        import json
        import random

        from qulacs import gate
        from qulacs.gate import (BitFlipNoise, DephasingNoise,
                                 DepolarizingNoise, IndependentXZNoise,
                                 Probabilistic, TwoQubitDepolarizingNoise, X,
                                 Y)

        r = random.random()
        gates = [
            Probabilistic([r, 1. - r], [X(0), Y(0)]),
            BitFlipNoise(0, random.random()), DephasingNoise(
                0, random.random()),
            IndependentXZNoise(0, random.random()), DepolarizingNoise(
                0, random.random()),
            TwoQubitDepolarizingNoise(0, 1, random.random(
            )),
        ]

        for g in gates:
            json_string = g.to_json()
            json.loads(json_string)
            g_json = gate.from_json(json_string)

            ds = g.get_distribution()
            ds_json = g_json.get_distribution()

            for i in range(len(ds)):
                self.assertAlmostEqual(ds[i], ds_json[i])

    def test_cptp_gate(self):
        import random

        from qulacs import QuantumState, gate
        from qulacs.gate import (CPTP, P0, P1, Adaptive, AmplitudeDampingNoise,
                                 Instrument, Measurement)

        n = 2
        gates = [
            AmplitudeDampingNoise(0, random.random()),
            CPTP([P0(0), P1(0)]),
            Instrument([P0(0), P1(0)], 0), Measurement(0, 0),
        ]

        for g in gates:
            g_list = g.get_gate_list()
            json_string = g.to_json()
            g_json = gate.from_json(json_string)
            g_json_list = g_json.get_gate_list()

            qs = QuantumState(n)

            for i in range(len(g_list)):
                gg = g_list[i]
                gg_json = g_json_list[i]
                self.assertEqual(gg.get_name(), gg_json.get_name())
                qs.set_Haar_random_state()
                qs_json = qs.copy()
                gg.update_quantum_state(qs)
                gg_json.update_quantum_state(qs_json)
                for i in range(n):
                    self.assertAlmostEqual(qs.get_zero_probability(
                        i), qs_json.get_zero_probability(i))

    def test_noisy_evolution_gate(self):
        import json

        from qulacs import (GeneralQuantumOperator, Observable, QuantumState,
                            gate)
        from qulacs.gate import (H, NoisyEvolution, NoisyEvolution_fast,
                                 PauliRotation)

        n = 2

        def execute_test_gate(is_fast):
            observable = Observable(n)
            observable.add_operator(1., "X 0")

            hamiltonian = Observable(n)
            hamiltonian.add_operator(1., "Z 0 Z 1")

            c_ops = []
            op = GeneralQuantumOperator(n)
            op.add_operator(0., "Z 0")
            c_ops.append(op)

            step = 10
            time = 3.14 / step
            dt = .001
            if is_fast:
                g = NoisyEvolution_fast(hamiltonian, c_ops, time)
            else:
                g = NoisyEvolution(hamiltonian, c_ops, time, dt)
            json_string = g.to_json()
            json.loads(json_string)
            g_json = gate.from_json(json_string)

            # reference gate
            g_ref = PauliRotation([0, 1], [3, 3], -time * 2)

            state = QuantumState(n)
            state_ref = QuantumState(n)
            h0 = H(0)
            h0.update_quantum_state(state)
            h0.update_quantum_state(state_ref)
            for k in range(step):
                g_json.update_quantum_state(state)
                g_ref.update_quantum_state(state_ref)
                exp = observable.get_expectation_value(state)
                exp_ref = observable.get_expectation_value(state_ref)
                self.assertAlmostEqual(exp.real, exp_ref.real)

        for _ in range(10):
            execute_test_gate(False)
            execute_test_gate(True)

    def test_circuit(self):
        import json

        from qulacs import QuantumCircuit, QuantumState, circuit
        from qulacs.gate import RandomUnitary

        n = 3

        circ = QuantumCircuit(n)
        for _ in range(3):
            g = RandomUnitary([0, 1, 2])
            circ.add_gate(g)

        json_string = circ.to_json()
        json.loads(json_string)
        circ_json = circuit.from_json(json_string)

        qs = QuantumState(n)
        qs.set_Haar_random_state()
        qs_json = qs.copy()

        circ.update_quantum_state(qs)
        circ_json.update_quantum_state(qs_json)

        for i in range(n):
            self.assertAlmostEqual(qs.get_zero_probability(
                i), qs_json.get_zero_probability(i))

    def test_parametric_circuit(self):
        import json
        import random

        from qulacs import ParametricQuantumCircuit, QuantumState, circuit
        from qulacs.gate import ParametricPauliRotation

        n = 2

        circ = ParametricQuantumCircuit(n)

        circ.add_parametric_RX_gate(0, random.random())
        circ.add_H_gate(1)
        circ.add_parametric_multi_Pauli_rotation_gate([0, 1], [1, 1], random.random())
        circ.add_CNOT_gate(0, 1)
        circ.add_parametric_RY_gate(1, random.random())
        circ.add_parametric_RZ_gate(0, random.random())

        json_string = circ.to_json()
        json.loads(json_string)
        circ_json = circuit.from_json(json_string)

        qs = QuantumState(n)
        qs.set_Haar_random_state()
        qs_json = qs.copy()

        circ.update_quantum_state(qs)
        circ_json.update_quantum_state(qs_json)

        for i in range(n):
            self.assertAlmostEqual(qs.get_zero_probability(
                i), qs_json.get_zero_probability(i))


class TestPickle(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_state_vector(self):
        import pickle

        import numpy

        from qulacs import QuantumState
        state = QuantumState(10)
        state.set_Haar_random_state()
        data = pickle.dumps(state)
        state2 = pickle.loads(data)
        assert isinstance(state2, QuantumState)
        assert numpy.allclose(state.get_vector(), state2.get_vector())
        
    def test_density_matrix(self):
        import pickle

        import numpy

        from qulacs import DensityMatrix
        state = DensityMatrix(5)
        state.set_Haar_random_state()
        data = pickle.dumps(state)
        state2 = pickle.loads(data)
        assert isinstance(state2, DensityMatrix)
        assert numpy.allclose(state.get_matrix(), state2.get_matrix())
        
    def test_quantum_circuit(self):
        import pickle

        import numpy as np

        from qulacs import QuantumCircuit
        from qulacs.gate import (CNOT, CZ, FREDKIN, RX, RY, RZ, SWAP, TOFFOLI,
                                 U1, U2, U3, DenseMatrix, H, Identity,
                                 RandomUnitary, S, Sdag, T, Tdag, X, Y, Z, add,
                                 merge, sqrtX, sqrtXdag, to_matrix_gate)
        gates = [
            Identity(0), X(0), Y(0), Z(0), H(0), S(0), Sdag(0), T(
                0), Tdag(0), sqrtX(0), sqrtXdag(0),
            CNOT(0, 1), CZ(0, 1), SWAP(0, 1), TOFFOLI(0, 1, 2), FREDKIN(
                0, 1, 2),
            DenseMatrix(0, np.eye(2)), DenseMatrix(
                [0, 1], np.eye(4)), RandomUnitary(
                [0, 1]), merge(
                X(0), Y(1)), add(X(0), Y(1)), to_matrix_gate(X(0)),
            U1(0, 0.), U2(0, 0., 0.), U3(
                0, 0., 0., 0.), RX(0, 0.), RY(0, 0.), RZ(0, 0.),
        ]
        circuit = QuantumCircuit(5)
        for x in gates:
            circuit.add_gate(x)
        data = pickle.dumps(circuit)
        del circuit
        circuit = pickle.loads(data)
        assert isinstance(circuit, QuantumCircuit)
        for x in range(circuit.get_gate_count()):
            assert np.allclose(circuit.get_gate(x).get_matrix(),
                               gates[x].get_matrix())

    def test_parametric_quantum_circuit(self):
        import pickle
        import random

        import numpy as np

        from qulacs import ParametricQuantumCircuit
        from qulacs.gate import ParametricPauliRotation

        gates = []
        circuit = ParametricQuantumCircuit(2)

        for _ in range(3):
            g = ParametricPauliRotation([0, 1], [1, 1], random.random())
            circuit.add_gate(g)
            gates.append(g)
            
        data = pickle.dumps(circuit)
        del circuit
        circuit = pickle.loads(data)
        assert isinstance(circuit, ParametricQuantumCircuit)
        for x in range(circuit.get_gate_count()):
            assert np.allclose(circuit.get_gate(x).get_matrix(),
                               gates[x].get_matrix())


if __name__ == "__main__":
    unittest.main()

