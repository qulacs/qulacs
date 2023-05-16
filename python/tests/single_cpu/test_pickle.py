import pickle
import random

import numpy
import numpy as np

from qulacs import (
    DensityMatrix,
    ParametricQuantumCircuit,
    QuantumCircuit,
    QuantumState,
)
from qulacs.gate import (
    CNOT,
    CZ,
    FREDKIN,
    RX,
    RY,
    RZ,
    SWAP,
    TOFFOLI,
    U1,
    U2,
    U3,
    DenseMatrix,
    H,
    Identity,
    ParametricPauliRotation,
    RandomUnitary,
    S,
    Sdag,
    T,
    Tdag,
    X,
    Y,
    Z,
    add,
    merge,
    sqrtX,
    sqrtXdag,
    to_matrix_gate,
)


class TestPickle:
    def test_state_vector(self) -> None:
        state = QuantumState(10)
        state.set_Haar_random_state()
        data = pickle.dumps(state)
        state2 = pickle.loads(data)
        assert isinstance(state2, QuantumState)
        assert numpy.allclose(state.get_vector(), state2.get_vector())

    def test_density_matrix(self) -> None:
        state = DensityMatrix(5)
        state.set_Haar_random_state()
        data = pickle.dumps(state)
        state2 = pickle.loads(data)
        assert isinstance(state2, DensityMatrix)
        assert numpy.allclose(state.get_matrix(), state2.get_matrix())

    def test_quantum_circuit(self) -> None:
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
            CNOT(0, 1),
            CZ(0, 1),
            SWAP(0, 1),
            TOFFOLI(0, 1, 2),
            FREDKIN(0, 1, 2),
            DenseMatrix(0, np.eye(2)),
            DenseMatrix([0, 1], np.eye(4)),
            RandomUnitary([0, 1]),
            merge(X(0), Y(1)),
            add(X(0), Y(1)),
            to_matrix_gate(X(0)),
            U1(0, 0.0),
            U2(0, 0.0, 0.0),
            U3(0, 0.0, 0.0, 0.0),
            RX(0, 0.0),
            RY(0, 0.0),
            RZ(0, 0.0),
        ]
        circuit = QuantumCircuit(5)
        for x in gates:
            circuit.add_gate(x)
        data = pickle.dumps(circuit)
        del circuit
        circuit = pickle.loads(data)
        assert isinstance(circuit, QuantumCircuit)
        for x in range(circuit.get_gate_count()):
            assert np.allclose(circuit.get_gate(x).get_matrix(), gates[x].get_matrix())

    def test_parametric_quantum_circuit(self) -> None:
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
            assert np.allclose(circuit.get_gate(x).get_matrix(), gates[x].get_matrix())
