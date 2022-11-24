import typing
from cmath import phase
from logging import NullHandler, getLogger
from typing import List

import numpy as np
import qulacs_core
from parse import parse, search
from qulacs import QuantumCircuit
from qulacs.gate import DenseMatrix, Identity

logger = getLogger(__name__)
logger.addHandler(NullHandler())


def convert_qulacs_circuit_to_QASM(cir: QuantumCircuit) -> typing.List[str]:
    # convert qulacs Quantum Circuit to QASM List[str].
    # This method uses extended QASM for serializing Density Matrix.
    # This serializes almost all gate type defined in qelib1.inc,
    # except for sqrtY gate.

    out_strs = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{cir.get_qubit_count()}];",
    ]

    for kai in range(cir.get_gate_count()):
        it = cir.get_gate(kai)
        clis = it.get_control_index_list()
        tlis = it.get_target_index_list()

        if it.get_name() == "CNOT":
            out_strs.append(f"cx q[{clis[0]}],q[{tlis[0]}];")
        elif it.get_name() == "CZ":
            out_strs.append(f"cz q[{clis[0]}],q[{tlis[0]}];")
        elif it.get_name() == "SWAP":
            out_strs.append(f"swap q[{tlis[0]}],q[{tlis[1]}];")
        elif it.get_name() == "Identity" or it.get_name() == "I":
            out_strs.append(f"id q[{tlis[0]}];")
        elif it.get_name() == "X":
            out_strs.append(f"x q[{tlis[0]}];")
        elif it.get_name() == "Y":
            out_strs.append(f"y q[{tlis[0]}];")
        elif it.get_name() == "Z":
            out_strs.append(f"z q[{tlis[0]}];")
        elif it.get_name() == "H":
            out_strs.append(f"h q[{tlis[0]}];")
        elif it.get_name() == "S":
            out_strs.append(f"s q[{tlis[0]}];")
        elif it.get_name() == "Sdag":
            out_strs.append(f"sdg q[{tlis[0]}];")
        elif it.get_name() == "T":
            out_strs.append(f"t q[{tlis[0]}];")
        elif it.get_name() == "Tdag":
            out_strs.append(f"tdg q[{tlis[0]}];")
        elif it.get_name() == "sqrtX":
            out_strs.append(f"sx q[{tlis[0]}];")
        elif it.get_name() == "sqrtXdag":
            out_strs.append(f"sxdg q[{tlis[0]}];")
        elif it.get_name() == "X-rotation":
            matrix = it.get_matrix()
            angle = phase(matrix[0][0] - matrix[1][0]) * 2
            if abs(angle) > 1e-5:
                out_strs.append(f"rx({angle}) q[{tlis[0]}];")
        elif it.get_name() == "Y-rotation":
            matrix = it.get_matrix()
            angle = phase(matrix[0][0] + matrix[1][0] * 1.0j) * 2
            if abs(angle) > 1e-5:
                out_strs.append(f"ry({angle}) q[{tlis[0]}];")
        elif it.get_name() == "Z-rotation":
            matrix = it.get_matrix()
            angle = phase(matrix[1][1] / matrix[0][0])
            if abs(angle) > 1e-5:
                out_strs.append(f"rz({angle}) q[{tlis[0]}];")
        elif isinstance(it, qulacs_core.QuantumGateMatrix):
            now_string = ""
            bit = len(it.get_target_index_list())
            matrix = it.get_matrix()
            now_string += "DenseMatrix("
            now_string += str(len(it.get_target_index_list()))
            now_string += ","
            now_string += str(len(it.get_control_index_list()))
            for i in range(2**bit):
                for j in range(2**bit):
                    now_string += f",{matrix[i][j].real:.6g},{matrix[i][j].imag:.6g}"
            for i in range(len(it.get_control_index_list())):
                if it.get_control_value_list()[i] == 1:
                    now_string += ",1"
                else:
                    now_string += ",0"

            now_string += ") "
            for aaa in it.get_target_index_list():
                if aaa == it.get_target_index_list()[0]:
                    now_string += f"q[{aaa}]"
                else:
                    now_string += f",q[{aaa}]"
            for aaa in it.get_control_index_list():
                now_string += f",q[{aaa}]"
            now_string += ";"
            out_strs.append(now_string)
        else:
            raise RuntimeError(f"unknown gate: {it.get_name()}")

    return out_strs


def convert_QASM_to_qulacs_circuit(
    input_strs: typing.List[str], *, remap_remove: bool = False
) -> QuantumCircuit:
    # convert QASM List[str] to qulacs QuantumCircuit.
    # constraints: qreg must be named q, and creg cannot be used.

    mapping: List[int] = []

    for instr_moto in input_strs:
        # process input string for parsing instruction.
        instr = instr_moto.lower().strip().replace(" ", "").replace("\t", "")

        if instr[0:4] == "qreg":
            ary = parse("qregq[{:d}];", instr)
            cir = QuantumCircuit(ary[0])
            if len(mapping) == 0:
                mapping = list(range(ary[0]))
        elif instr[0:2] == "cx":
            ary = parse("cxq[{:d}],q[{:d}];", instr)
            cir.add_CNOT_gate(mapping[ary[0]], mapping[ary[1]])
        elif instr[0:2] == "cz":
            ary = parse("czq[{:d}],q[{:d}];", instr)
            cir.add_CZ_gate(mapping[ary[0]], mapping[ary[1]])
        elif instr[0:4] == "swap":
            ary = parse("swapq[{:d}],q[{:d}];", instr)
            cir.add_SWAP_gate(mapping[ary[0]], mapping[ary[1]])
        elif instr[0:2] == "id":
            ary = parse("idq[{:d}];", instr)
            cir.add_gate(Identity(mapping[ary[0]]))
        elif instr[0:2] == "xq":
            ary = parse("xq[{:d}];", instr)
            cir.add_X_gate(mapping[ary[0]])
        elif instr[0:2] == "yq":
            ary = parse("yq[{:d}];", instr)
            cir.add_Y_gate(mapping[ary[0]])
        elif instr[0:2] == "zq":
            ary = parse("zq[{:d}];", instr)
            cir.add_Z_gate(mapping[ary[0]])
        elif instr[0:2] == "hq":
            ary = parse("hq[{:d}];", instr)
            cir.add_H_gate(mapping[ary[0]])
        elif instr[0:2] == "sq":
            ary = parse("sq[{:d}];", instr)
            cir.add_S_gate(mapping[ary[0]])
        elif instr[0:4] == "sdgq":
            ary = parse("sdgq[{:d}];", instr)
            cir.add_Sdag_gate(mapping[ary[0]])
        elif instr[0:2] == "tq":
            ary = parse("tq[{:d}];", instr)
            cir.add_T_gate(mapping[ary[0]])
        elif instr[0:4] == "tdgq":
            ary = parse("tdgq[{:d}];", instr)
            cir.add_Tdag_gate(mapping[ary[0]])
        elif instr[0:2] == "rx":
            ary = parse("rx({:g})q[{:d}];", instr)
            cir.add_RX_gate(mapping[ary[1]], -ary[0])
        elif instr[0:2] == "ry":
            ary = parse("ry({:g})q[{:d}];", instr)
            cir.add_RY_gate(mapping[ary[1]], -ary[0])
        elif instr[0:2] == "rz":
            ary = parse("rz({:g})q[{:d}];", instr)
            cir.add_RZ_gate(mapping[ary[1]], -ary[0])
        elif instr[0:1] == "p":
            ary = parse("p({:g})q[{:d}];", instr)
            cir.add_U1_gate(mapping[ary[1]], ary[0])
        elif instr[0:2] == "u1":
            ary = parse("u1({:g})q[{:d}];", instr)
            cir.add_U1_gate(mapping[ary[1]], ary[0])
        elif instr[0:2] == "u2":
            ary = parse("u2({:g},{:g})q[{:d}];", instr)
            cir.add_U2_gate(mapping[ary[2]], ary[0], ary[1])
        elif instr[0:2] == "u3":
            ary = parse("u3({:g},{:g},{:g})q[{:d}];", instr)
            cir.add_U3_gate(mapping[ary[3]], ary[0], ary[1], ary[2])
        elif instr[0:1] == "u":
            ary = parse("u({:g},{:g},{:g})q[{:d}];", instr)
            cir.add_U3_gate(mapping[ary[3]], ary[0], ary[1], ary[2])
        elif instr[0:3] == "sxq":
            ary = parse("sxq[{:d}];", instr)
            cir.add_sqrtX_gate(mapping[ary[0]])
        elif instr[0:5] == "sxdgq":
            ary = parse("sxdgq[{:d}];", instr)
            cir.add_sqrtXdag_gate(mapping[ary[0]])
        elif instr[0:11] == "densematrix":
            ary = search("densematrix({:d},{:d}", instr)
            parsestr = "densematrix({:d},{:d}"
            for i in range(4 ** ary[0]):
                parsestr += ",{:g},{:g}"
            for i in range(ary[1]):
                parsestr += ",{:d}"
            parsestr += ")q[{:d}]"
            for i in range(ary[0] + ary[1] - 1):
                parsestr += ",q[{:d}]"
            parsestr += ";"
            deary = parse(parsestr, instr)
            gate_mat = np.zeros((2 ** ary[0], 2 ** ary[0]), dtype="complex")
            bas = 2
            for i in range(2 ** ary[0]):
                for j in range(2 ** ary[0]):
                    gate_mat[i][j] = deary[bas] + deary[bas + 1] * 1.0j
                    bas += 2
            control_values = []
            for i in range(ary[1]):
                control_values.append(mapping[deary[bas]])
                bas += 1
            terget_indexes = []
            for i in range(ary[0]):
                terget_indexes.append(mapping[deary[bas]])
                bas += 1

            dense_gate = DenseMatrix(terget_indexes, gate_mat)  # type:ignore
            for i in range(ary[1]):
                control_index = deary[bas]
                bas += 1
                dense_gate.add_control_qubit(control_index, control_values[i])
            cir.add_gate(dense_gate)
        elif remap_remove and instr[0:4] == "//q[":
            ary = parse("//q[{:d}]-->q[{:d}]", instr)
            if not (ary is None):
                mapping[ary[0]] = ary[1]
        elif remap_remove and instr[0:8] == "//qubits":
            ary = parse("//qubits:{:d}", instr)
            mapping = list(range(ary[0]))
        elif instr == "openqasm2.0;" or instr == 'include"qelib1.inc";':
            # related to file format, not for read.
            pass
        else:
            raise RuntimeError(f"unknown line: {instr}")
    return cir
