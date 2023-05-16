import json
import random

import numpy as np
import pytest
from scipy.sparse import csc_matrix, lil_matrix

from qulacs import (
    GeneralQuantumOperator,
    Observable,
    ParametricQuantumCircuit,
    PauliOperator,
    QuantumCircuit,
    QuantumState,
    circuit,
    gate,
    observable,
    quantum_operator,
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
    AmplitudeDampingNoise,
    BitFlipNoise,
    DenseMatrix,
    DephasingNoise,
    DepolarizingNoise,
    H,
    Identity,
    IndependentXZNoise,
    Instrument,
    Measurement,
    NoisyEvolution,
    NoisyEvolution_fast,
    ParametricPauliRotation,
    ParametricRX,
    ParametricRY,
    ParametricRZ,
    Pauli,
    PauliRotation,
    Probabilistic,
    RandomUnitary,
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


class TestJSON:
    def test_operator(self) -> None:
        n = 5

        def random_pauli_operator() -> PauliOperator:
            op = PauliOperator(
                (random.random() * 2 - 1) + (random.random() * 2 - 1) * 1j
            )
            for _ in range(random.randint(1, 5)):
                op.add_single_Pauli(random.randint(0, n - 1), random.randint(0, 3))
            return op

        def random_hermitian_pauli_operator() -> PauliOperator:
            op = PauliOperator(random.random() * 2 - 1)
            for _ in range(random.randint(1, 5)):
                op.add_single_Pauli(random.randint(0, n - 1), random.randint(0, 3))
            return op

        original_operator = GeneralQuantumOperator(5)
        for _ in range(5):
            original_operator.add_operator(random_pauli_operator())

        json_string = original_operator.to_json()
        json.loads(json_string)
        restored_operator = quantum_operator.from_json(json_string)

        for _ in range(3):
            state = QuantumState(n)
            assert original_operator.get_expectation_value(state) == pytest.approx(
                restored_operator.get_expectation_value(state)
            )

        original_observable = Observable(n)
        for _ in range(5):
            original_observable.add_operator(random_hermitian_pauli_operator())

        json_string = original_observable.to_json()
        json.loads(json_string)
        restored_observable = observable.from_json(json_string)

        for _ in range(3):
            state = QuantumState(n)
            state.set_Haar_random_state()
            assert original_observable.get_expectation_value(state) == pytest.approx(
                restored_observable.get_expectation_value(state)
            )

        non_hermitian_operator = GeneralQuantumOperator(1)
        non_hermitian_operator.add_operator(1j, "X 0")
        with pytest.raises(RuntimeError):
            observable.from_json(non_hermitian_operator.to_json())

    def test_gate(self) -> None:
        n = 3

        def execute_test_gate() -> None:
            qs = QuantumState(n)
            sparse_mat = lil_matrix((4, 4))
            sparse_mat[0, 0] = 1
            sparse_mat[1, 1] = 1

            axis = QuantumState(n)
            axis.set_Haar_random_state()

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
                CNOT(0, 1),
                CZ(0, 1),
                SWAP(0, 1),
                TOFFOLI(0, 1, 2),
                FREDKIN(0, 1, 2),
                Pauli([0, 1], [1, 2]),
                PauliRotation([0, 1], [1, 2], random.random()),
                StateReflection(axis),
                merge(X(0), Y(1)),
                add(X(0), Y(1)),
                to_matrix_gate(X(0)),
                P0(0),
                P1(0),
                U1(0, random.random()),
                U2(0, random.random(), random.random()),
                U3(0, random.random(), random.random(), random.random()),
                RX(0, random.random()),
                RY(0, random.random()),
                RZ(0, random.random()),
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
                    assert qs.get_zero_probability(i) == pytest.approx(
                        qs_json.get_zero_probability(i)
                    )

        for _ in range(10):
            execute_test_gate()

    def test_parametric_gate(self) -> None:
        n = 3
        qs = QuantumState(n)

        gates = [
            ParametricRX(0, random.random()),
            ParametricRY(0, random.random()),
            ParametricRZ(0, random.random()),
            ParametricPauliRotation([0, 1], [1, 1], random.random()),
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
                assert qs.get_zero_probability(i) == pytest.approx(
                    qs_json.get_zero_probability(i)
                )

    def test_matrix_gate(self) -> None:
        n = 3

        def execute_test_matrix_gate() -> None:
            qs = QuantumState(n)

            # DenseMatrix
            matrix = np.array(
                [[random.random(), random.random()], [random.random(), random.random()]]
            )
            g = DenseMatrix(0, matrix)
            qs.set_Haar_random_state()
            qs_json = qs.copy()
            g.update_quantum_state(qs)
            json_string = g.to_json()
            json.loads(json_string)
            g_json = gate.from_json(json_string)
            g_json.update_quantum_state(qs_json)
            for i in range(n):
                assert qs.get_zero_probability(i) == pytest.approx(
                    qs_json.get_zero_probability(i)
                )

            # SparseMatrix
            matrix = np.array(
                [[random.random(), random.random()], [random.random(), random.random()]]
            )
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
                assert qs.get_zero_probability(i) == pytest.approx(
                    qs_json.get_zero_probability(i)
                )

        for _ in range(10):
            execute_test_matrix_gate()

    def test_probabilistic_gate(self) -> None:
        r = random.random()
        gates = [
            Probabilistic([r, 1.0 - r], [X(0), Y(0)]),
            BitFlipNoise(0, random.random()),
            DephasingNoise(0, random.random()),
            IndependentXZNoise(0, random.random()),
            DepolarizingNoise(0, random.random()),
            TwoQubitDepolarizingNoise(0, 1, random.random()),
        ]

        for g in gates:
            json_string = g.to_json()
            json.loads(json_string)
            g_json = gate.from_json(json_string)

            ds = g.get_distribution()
            ds_json = g_json.get_distribution()

            for i in range(len(ds)):
                assert ds[i] == pytest.approx(ds_json[i])

    def test_cptp_gate(self) -> None:
        n = 2
        gates = [
            AmplitudeDampingNoise(0, random.random()),
            CPTP([P0(0), P1(0)]),
            Instrument([P0(0), P1(0)], 0),
            Measurement(0, 0),
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
                assert gg.get_name() == gg_json.get_name()
                qs.set_Haar_random_state()
                qs_json = qs.copy()
                gg.update_quantum_state(qs)
                gg_json.update_quantum_state(qs_json)
                for i in range(n):
                    assert qs.get_zero_probability(i) == pytest.approx(
                        qs_json.get_zero_probability(i)
                    )

    def test_noisy_evolution_gate(self) -> None:
        n = 2

        def execute_test_gate(is_fast) -> None:
            observable = Observable(n)
            observable.add_operator(1.0, "X 0")

            hamiltonian = Observable(n)
            hamiltonian.add_operator(1.0, "Z 0 Z 1")

            c_ops = []
            op = GeneralQuantumOperator(n)
            op.add_operator(0.0, "Z 0")
            c_ops.append(op)

            step = 10
            time = 3.14 / step
            dt = 0.001
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
                assert exp.real == pytest.approx(exp_ref.real)

        for _ in range(10):
            execute_test_gate(False)
            execute_test_gate(True)

    def test_circuit(self) -> None:
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
            assert qs.get_zero_probability(i) == pytest.approx(
                qs_json.get_zero_probability(i)
            )

    def test_parametric_circuit(self) -> None:
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
            assert qs.get_zero_probability(i) == pytest.approx(
                qs_json.get_zero_probability(i)
            )
