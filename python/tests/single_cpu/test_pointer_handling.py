import numpy as np
import pytest
from scipy.sparse import lil_matrix

from qulacs import (
    DensityMatrix,
    Observable,
    ParametricQuantumCircuit,
    QuantumCircuit,
    QuantumState,
    StateVector,
)
from qulacs.gate import (
    CNOT,
    CPTP,
    CZ,
    FREDKIN,
    P0,
    P1,
    RX,
    RY,
    RZ,
    SWAP,
    TOFFOLI,
    U1,
    U2,
    U3,
    Adaptive,
    AmplitudeDampingNoise,
    BitFlipNoise,
    DenseMatrix,
    DephasingNoise,
    DepolarizingNoise,
    DiagonalMatrix,
    H,
    Identity,
    IndependentXZNoise,
    Instrument,
    Measurement,
    ParametricPauliRotation,
    ParametricRX,
    ParametricRY,
    ParametricRZ,
    Pauli,
    PauliRotation,
    Probabilistic,
    RandomUnitary,
    ReversibleBoolean,
    S,
    Sdag,
    SparseMatrix,
    StateReflection,
    T,
    Tdag,
    TwoQubitDepolarizingNoise,
    X,
    Y,
    Z,
    add,
    merge,
    sqrtX,
    sqrtXdag,
    sqrtY,
    sqrtYdag,
    to_matrix_gate,
)
from qulacs.state import partial_trace, permutate_qubit, tensor_product


class TestPointerHandling:
    def test_pointer_del(self) -> None:
        qc = QuantumCircuit(1)
        gate = X(0)
        qc.add_gate(gate)
        del gate
        del qc

    def test_internal_return_value_of_get_gate_is_valid(self) -> None:
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

        def func2() -> None:
            qs = QuantumState(2)
            circuit = func()
            circuit.update_quantum_state(qs)

        func2()

    def test_circuit_add_gate(self) -> None:
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
            Identity(0),
            X(0),
            Y(0),
            Z(0),
            H(0),
            S(0),
            Sdag(0),
            T(0),
            Tdag(0),
            sqrtX(0),
            sqrtXdag(0),
            sqrtY(0),
            sqrtYdag(0),
            Probabilistic([0.5, 0.5], [X(0), Y(0)]),
            CPTP([P0(0), P1(0)]),
            Instrument([P0(0), P1(0)], 1),
            Adaptive(X(0), adap),
            CNOT(0, 1),
            CZ(0, 1),
            SWAP(0, 1),
            TOFFOLI(0, 1, 2),
            FREDKIN(0, 1, 2),
            Pauli([0, 1], [1, 2]),
            PauliRotation([0, 1], [1, 2], 0.1),
            DenseMatrix(0, np.eye(2)),
            DenseMatrix([0, 1], np.eye(4)),
            SparseMatrix([0, 1], sparse_mat),
            DiagonalMatrix([0, 1], np.ones(4)),
            RandomUnitary([0, 1]),
            ReversibleBoolean([0, 1], func),
            StateReflection(ref),
            BitFlipNoise(0, 0.1),
            DephasingNoise(0, 0.1),
            IndependentXZNoise(0, 0.1),
            DepolarizingNoise(0, 0.1),
            TwoQubitDepolarizingNoise(0, 1, 0.1),
            AmplitudeDampingNoise(0, 0.1),
            Measurement(0, 1),
            merge(X(0), Y(1)),
            add(X(0), Y(1)),
            to_matrix_gate(X(0)),
            P0(0),
            P1(0),
            U1(0, 0.0),
            U2(0, 0.0, 0.0),
            U3(0, 0.0, 0.0, 0.0),
            RX(0, 0.0),
            RY(0, 0.0),
            RZ(0, 0.0),
        ]
        gates.append(merge(gates[0], gates[1]))
        gates.append(add(gates[0], gates[1]))

        ref = None  # type: ignore
        for gate in gates:
            qc.add_gate(gate)

        for gate in gates:
            qc.add_gate(gate)

        qc.update_quantum_state(qs)
        qc = None  # type: ignore
        qs = None  # type: ignore
        for gate in gates:
            gate = None

        gates = None  # type: ignore

    def test_circuit_add_parametric_gate(self) -> None:
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
            Identity(0),
            X(0),
            Y(0),
            Z(0),
            H(0),
            S(0),
            Sdag(0),
            T(0),
            Tdag(0),
            sqrtX(0),
            sqrtXdag(0),
            sqrtY(0),
            sqrtYdag(0),
            Probabilistic([0.5, 0.5], [X(0), Y(0)]),
            CPTP([P0(0), P1(0)]),
            Instrument([P0(0), P1(0)], 1),
            Adaptive(X(0), adap),
            CNOT(0, 1),
            CZ(0, 1),
            SWAP(0, 1),
            TOFFOLI(0, 1, 2),
            FREDKIN(0, 1, 2),
            Pauli([0, 1], [1, 2]),
            PauliRotation([0, 1], [1, 2], 0.1),
            DenseMatrix(0, np.eye(2)),
            DenseMatrix([0, 1], np.eye(4)),
            SparseMatrix([0, 1], sparse_mat),
            DiagonalMatrix([0, 1], np.ones(4)),
            RandomUnitary([0, 1]),
            ReversibleBoolean([0, 1], func),
            StateReflection(ref),
            BitFlipNoise(0, 0.1),
            DephasingNoise(0, 0.1),
            IndependentXZNoise(0, 0.1),
            DepolarizingNoise(0, 0.1),
            TwoQubitDepolarizingNoise(0, 1, 0.1),
            AmplitudeDampingNoise(0, 0.1),
            Measurement(0, 1),
            merge(X(0), Y(1)),
            add(X(0), Y(1)),
            to_matrix_gate(X(0)),
            P0(0),
            P1(0),
            U1(0, 0.0),
            U2(0, 0.0, 0.0),
            U3(0, 0.0, 0.0, 0.0),
            RX(0, 0.0),
            RY(0, 0.0),
            RZ(0, 0.0),
        ]

        gates.append(merge(gates[0], gates[1]))
        gates.append(add(gates[0], gates[1]))

        parametric_gates = [
            ParametricRX(0, 0.1),
            ParametricRY(0, 0.1),
            ParametricRZ(0, 0.1),
            ParametricPauliRotation([0, 1], [1, 1], 0.1),
        ]

        ref = None  # type: ignore
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
        qc = None  # type: ignore
        qs = None  # type: ignore
        for gate in gates:
            gate = None
        for pgate in parametric_gates:
            gate = None

        gates = None  # type: ignore
        parametric_gates = None  # type: ignore

    def test_add_same_gate_multiple_time(self) -> None:
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

    def test_observable(self) -> None:
        obs = Observable(1)
        obs.add_operator(1.0, "X 0")
        term = obs.get_term(0)
        del term

    def test_add_gate(self) -> None:
        circuit = QuantumCircuit(1)
        gate = X(0)
        circuit.add_gate(gate)
        del gate
        s = circuit.to_string()  # noqa
        del circuit

    def test_add_gate_in_parametric_circuit(self) -> None:
        circuit = ParametricQuantumCircuit(1)
        gate = X(0)
        circuit.add_gate(gate)
        del gate
        s = circuit.to_string()  # noqa
        del circuit

    def test_state_reflection(self) -> None:
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

    def test_sparse_matrix(self) -> None:
        n = 5
        state = QuantumState(n)
        matrix = lil_matrix((4, 4), dtype=np.complex128)
        matrix[0, 0] = 1 + 1.0j
        matrix[1, 1] = 1.0 + 1.0j
        gate = SparseMatrix([0, 1], matrix)
        gate.update_quantum_state(state)
        del gate
        del state

    def test_copied_parametric_gate(self) -> None:
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
            assert val == pytest.approx(gc * 0.1 + 0.1), "check vector size"

        d = c.copy()
        del c
        for gc in range(d.get_parameter_count()):
            val = d.get_parameter(gc)
            d.set_parameter(gc, val + 10)
            val = d.get_parameter(gc)
            assert val == pytest.approx(11.1 + gc * 0.1), "check vector size"

        qs = QuantumState(1)
        d.update_quantum_state(qs)
        del d
        del qs

    def test_parametric_gate_position(self) -> None:
        def check(pqc, idlist):
            cnt = pqc.get_parameter_count()
            assert cnt == len(idlist)
            for ind in range(cnt):
                pos = pqc.get_parametric_gate_position(ind)
                assert pos == idlist[ind]

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


class TestDensityMatrixHandling:
    def test_density_matrix(self) -> None:
        num_qubit = 5
        sv = StateVector(num_qubit)
        dm = DensityMatrix(num_qubit)
        sv.set_Haar_random_state(seed=0)
        dm.load(sv)
        svv = np.atleast_2d(sv.get_vector()).T
        mat = np.dot(svv, svv.T.conj())
        assert np.allclose(dm.get_matrix(), mat), "check pure matrix to density matrix"

    def test_tensor_product_sv(self) -> None:
        num_qubit = 4
        sv1 = StateVector(num_qubit)
        sv2 = StateVector(num_qubit)
        sv1.set_Haar_random_state(seed=0)
        sv2.set_Haar_random_state(seed=1)
        sv3 = tensor_product(sv1, sv2)
        sv3_test = np.kron(sv1.get_vector(), sv2.get_vector())
        assert np.allclose(
            sv3_test, sv3.get_vector()
        ), "check pure state tensor product"
        del sv1
        del sv2
        del sv3

    def test_tensor_product_dm(self) -> None:
        num_qubit = 4
        dm1 = DensityMatrix(num_qubit)
        dm2 = DensityMatrix(num_qubit)
        dm1.set_Haar_random_state(seed=0)
        dm2.set_Haar_random_state(seed=1)
        dm3 = tensor_product(dm1, dm2)
        dm3_test = np.kron(dm1.get_matrix(), dm2.get_matrix())

        assert np.allclose(
            dm3_test, dm3.get_matrix()
        ), "check density matrix tensor product"

        del dm1
        del dm2
        del dm3

    def test_tensor_product_different_size_sv(self) -> None:
        num_qubit = 4
        sv1 = StateVector(num_qubit)
        sv2 = StateVector(num_qubit + 1)
        sv1.set_Haar_random_state(seed=0)
        sv2.set_Haar_random_state(seed=1)
        sv3 = tensor_product(sv1, sv2)
        sv3_test = np.kron(sv1.get_vector(), sv2.get_vector())
        assert np.allclose(
            sv3_test, sv3.get_vector()
        ), "check pure state tensor product"
        del sv1
        del sv2
        del sv3

    def test_tensor_product_different_size_dm(self) -> None:
        num_qubit = 4
        dm1 = DensityMatrix(num_qubit)
        dm2 = DensityMatrix(num_qubit + 1)
        dm1.set_Haar_random_state(seed=0)
        dm2.set_Haar_random_state(seed=1)
        dm3 = tensor_product(dm1, dm2)
        dm3_test = np.kron(dm1.get_matrix(), dm2.get_matrix())
        assert np.allclose(
            dm3_test, dm3.get_matrix()
        ), "check density matrix tensor product"
        del dm1
        del dm2
        del dm3

    def test_permutate_qubit_sv(self) -> None:
        num_qubit = 8
        sv = StateVector(num_qubit)
        sv.set_Haar_random_state(seed=0)
        order = np.arange(num_qubit)
        np.random.shuffle(order)

        arr = []
        for ind in range(2**num_qubit):
            s = format(ind, "0{}b".format(num_qubit))
            s = np.array(list(s[::-1]))  # type: ignore
            v = np.array(["*"] * num_qubit)
            for ind in range(len(s)):
                v[order[ind]] = s[ind]
            s = ("".join(v))[::-1]
            arr.append(int(s, 2))

        sv_perm = permutate_qubit(sv, order)  # type: ignore
        assert np.allclose(
            sv.get_vector()[arr], sv_perm.get_vector()
        ), "check pure state permutation"
        del sv_perm
        del sv

    def test_permutate_qubit_dm(self) -> None:
        num_qubit = 3
        dm = DensityMatrix(num_qubit)
        dm.set_Haar_random_state(seed=0)
        order = np.arange(num_qubit)
        np.random.shuffle(order)

        arr = []
        for ind in range(2**num_qubit):
            s = format(ind, "0{}b".format(num_qubit))
            s = np.array(list(s[::-1]))  # type: ignore
            v = np.array(["*"] * num_qubit)
            for ind in range(len(s)):
                v[order[ind]] = s[ind]
            s = ("".join(v))[::-1]
            arr.append(int(s, 2))

        dm_perm = permutate_qubit(dm, order)  # type: ignore
        dm_perm_test = dm.get_matrix()
        dm_perm_test = dm_perm_test[arr, :]
        dm_perm_test = dm_perm_test[:, arr]
        assert np.allclose(
            dm_perm_test, dm_perm.get_matrix()
        ), "check density matrix permutation"
        del dm_perm
        del dm

    def test_partial_trace_dm(self) -> None:
        num_qubit = 5
        num_traceout = 2
        dm = DensityMatrix(num_qubit)
        dm.set_Haar_random_state(seed=0)
        mat = dm.get_matrix()

        target = np.arange(num_qubit)
        np.random.shuffle(target)
        target = target[:num_traceout]
        target_cor = [num_qubit - 1 - i for i in target]
        target_cor.sort()

        dmt = mat.reshape([2, 2] * num_qubit)
        for cnt, val in enumerate(target_cor):
            ofs = num_qubit - cnt
            dmt = np.trace(dmt, axis1=val - cnt, axis2=ofs + val - cnt)
        dmt = dmt.reshape(
            [2 ** (num_qubit - num_traceout), 2 ** (num_qubit - num_traceout)]
        )

        pdm = partial_trace(dm, target)  # type: ignore
        assert np.allclose(pdm.get_matrix(), dmt), "check density matrix partial trace"
        del dm, pdm

    def test_partial_trace_sv(self) -> None:
        num_qubit = 6
        num_traceout = 4
        sv = StateVector(num_qubit)
        sv.set_Haar_random_state(seed=0)
        svv = np.atleast_2d(sv.get_vector()).T
        mat = np.dot(svv, svv.T.conj())

        target = np.arange(num_qubit)
        np.random.shuffle(target)
        target = target[:num_traceout]
        target_cor = [num_qubit - 1 - i for i in target]
        target_cor.sort()

        dmt = mat.reshape([2, 2] * num_qubit)
        for cnt, val in enumerate(target_cor):
            ofs = num_qubit - cnt
            dmt = np.trace(dmt, axis1=val - cnt, axis2=ofs + val - cnt)
        dmt = dmt.reshape(
            [2 ** (num_qubit - num_traceout), 2 ** (num_qubit - num_traceout)]
        )

        pdm = partial_trace(sv, target)  # type: ignore
        assert np.allclose(pdm.get_matrix(), dmt), "check pure state partial trace"
