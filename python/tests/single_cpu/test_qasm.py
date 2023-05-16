import textwrap

import numpy as np

from qulacs import QuantumCircuit
from qulacs.converter import (
    convert_QASM_to_qulacs_circuit,
    convert_qulacs_circuit_to_QASM,
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


class TestQASM:
    def test_qasm_converter(self) -> None:
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
        QASM_strs = convert_qulacs_circuit_to_QASM(circuit)
        transpiled_circuit = convert_QASM_to_qulacs_circuit(QASM_strs)
        for x in range(transpiled_circuit.get_gate_count()):
            assert np.allclose(
                transpiled_circuit.get_gate(x).get_matrix(), gates[x].get_matrix()
            )

    def test_qasm_converter_with_signed_params(self) -> None:
        angle = np.pi / 4.0
        gates = [
            # backward rotation
            U1(0, angle),
            U2(0, angle, angle),
            U3(0, angle, angle, angle),
            RX(0, angle),
            RY(0, angle),
            RZ(0, angle),
            # forward rotation
            U1(0, -angle),
            U2(0, -angle, -angle),
            U3(0, -angle, -angle, -angle),
            RX(0, -angle),
            RY(0, -angle),
            RZ(0, -angle),
        ]
        circuit = QuantumCircuit(5)
        for x in gates:
            circuit.add_gate(x)
        QASM_strs = convert_qulacs_circuit_to_QASM(circuit)
        transpiled_circuit = convert_QASM_to_qulacs_circuit(QASM_strs)
        for x in range(transpiled_circuit.get_gate_count()):
            assert np.allclose(
                transpiled_circuit.get_gate(x).get_matrix(), gates[x].get_matrix()
            )

    def test_qasm_converter_from_qasm_str(self) -> None:
        # equals RX(1, np.pi / 4.0)
        qasm = textwrap.dedent(
            """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[4];
            rx(-0.785398163397448) q[1];
            """
        ).strip()

        circuit = convert_QASM_to_qulacs_circuit(qasm.splitlines())
        recoverd = convert_qulacs_circuit_to_QASM(circuit)
        assert qasm == "\n".join(recoverd)
