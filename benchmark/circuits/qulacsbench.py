"""
build a Qulacs benchmark Circuit with parameter
"""

import numpy as np

from qulacs import QuantumCircuit


def get_act_idx(i, local_qc, global_qc):
    if swapped:
        if i >= local_qc:
            return i - global_qc
        elif i >= local_qc - global_qc:
            return i + global_qc
        else:
            return i
    else:
        return i


def first_rotation(circuit, nqubits, global_qc):
    global swapped
    local_qc = nqubits - global_qc

    local_qubit_list = list(
        filter(lambda i: get_act_idx(i, local_qc, global_qc) < local_qc, range(nqubits))
    )
    global_qubit_list = list(
        filter(
            lambda i: get_act_idx(i, local_qc, global_qc) >= local_qc, range(nqubits)
        )
    )

    for k in local_qubit_list:
        k_phy = get_act_idx(k, local_qc, global_qc)
        if _print_debug and mpirank == 0:
            print("RX/RZ {} ({})".format(k, k_phy))
        circuit.add_RX_gate(k_phy, np.random.rand())
        circuit.add_RZ_gate(k_phy, np.random.rand())

    if global_qc > 0:
        if _print_debug and mpirank == 0:
            print(
                "FusedSWAP {} {} {}".format(local_qc - global_qc, local_qc, global_qc)
            )
        circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)
        swapped = not swapped

    for k in global_qubit_list:
        k_phy = get_act_idx(k, local_qc, global_qc)
        if _print_debug and mpirank == 0:
            print("RX/RZ {} ({})".format(k, k_phy))
        circuit.add_RX_gate(k_phy, np.random.rand())
        circuit.add_RZ_gate(k_phy, np.random.rand())


def mid_rotation(circuit, nqubits, global_qc):
    global swapped
    local_qc = nqubits - global_qc

    local_qubit_list = list(
        filter(lambda i: get_act_idx(i, local_qc, global_qc) < local_qc, range(nqubits))
    )
    global_qubit_list = list(
        filter(
            lambda i: get_act_idx(i, local_qc, global_qc) >= local_qc, range(nqubits)
        )
    )

    for k in local_qubit_list:
        k_phy = get_act_idx(k, local_qc, global_qc)
        if _print_debug and mpirank == 0:
            print("RZ/RX/RZ {} ({})".format(k, k_phy))
        circuit.add_RZ_gate(k_phy, np.random.rand())
        circuit.add_RX_gate(k_phy, np.random.rand())
        circuit.add_RZ_gate(k_phy, np.random.rand())

    if global_qc > 0:
        if _print_debug and mpirank == 0:
            print(
                "FusedSWAP {} {} {}".format(local_qc - global_qc, local_qc, global_qc)
            )
        circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)
        swapped = not swapped

    for k in global_qubit_list:
        k_phy = get_act_idx(k, local_qc, global_qc)
        if _print_debug and mpirank == 0:
            print("RZ/RX/RZ {} ({})".format(k, k_phy))
        circuit.add_RZ_gate(k_phy, np.random.rand())
        circuit.add_RX_gate(k_phy, np.random.rand())
        circuit.add_RZ_gate(k_phy, np.random.rand())


def last_rotation(circuit, nqubits, global_qc):
    global swapped
    local_qc = nqubits - global_qc

    local_qubit_list = list(
        filter(lambda i: get_act_idx(i, local_qc, global_qc) < local_qc, range(nqubits))
    )
    global_qubit_list = list(
        filter(
            lambda i: get_act_idx(i, local_qc, global_qc) >= local_qc, range(nqubits)
        )
    )

    for k in local_qubit_list:
        k_phy = get_act_idx(k, local_qc, global_qc)
        if _print_debug and mpirank == 0:
            print("RZ/RX {} ({})".format(k, k_phy))
        circuit.add_RZ_gate(k_phy, np.random.rand())
        circuit.add_RX_gate(k_phy, np.random.rand())

    if global_qc > 0:
        if _print_debug and mpirank == 0:
            print(
                "FusedSWAP {} {} {}".format(local_qc - global_qc, local_qc, global_qc)
            )
        circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)
        swapped = not swapped

    for k in global_qubit_list:
        k_phy = get_act_idx(k, local_qc, global_qc)
        if _print_debug and mpirank == 0:
            print("RZ/RX {} ({})".format(k, k_phy))
        circuit.add_RZ_gate(k_phy, np.random.rand())
        circuit.add_RX_gate(k_phy, np.random.rand())


def entangler(circuit, nqubits, pairs, global_qc):
    global swapped
    local_qc = nqubits - global_qc

    for a, b in pairs:
        if global_qc > 0 and get_act_idx(b, local_qc, global_qc) >= local_qc:
            if _print_debug and mpirank == 0:
                print(
                    "FusedSWAP {} {} {}".format(
                        local_qc - global_qc, local_qc, global_qc
                    )
                )
            circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)
            swapped = not swapped

        a_phy = get_act_idx(a, local_qc, global_qc)
        b_phy = get_act_idx(b, local_qc, global_qc)
        if _print_debug and mpirank == 0:
            print("CNOT {} {} ({} {})".format(a, b, a_phy, b_phy))
        circuit.add_CNOT_gate(a_phy, b_phy)


def build_circuit(nqubits, global_nqubits, depth=9, verbose=False, random_gen=""):
    use_fusedswap = True if global_nqubits > 0 else False
    local_nqubits = nqubits - global_nqubits
    if random_gen == "":
        rng = np.random.default_rng()
    else:
        rng = random_gen

    circuit = QuantumCircuit(nqubits)
    pairs = [(i, (i + 1) % nqubits) for i in range(nqubits)]
    global_qc = global_nqubits
    local_qc = nqubits - global_qc

    seed = rng.integers(10000)
    np.random.seed(seed)
    global _print_debug
    global swapped
    _print_debug = verbose
    swapped = False

    # add gates
    first_rotation(circuit, nqubits, global_qc)
    entangler(circuit, nqubits, pairs, global_qc)
    for k in range(depth):
        mid_rotation(circuit, nqubits, global_qc)
        entangler(circuit, nqubits, pairs, global_qc)
    last_rotation(circuit, nqubits, global_qc)

    # recover if needed
    if use_fusedswap and swapped:
        if _print_debug and mpirank == 0:
            print(
                "FusedSWAP {} {} {}".format(local_qc - global_qc, local_qc, global_qc)
            )
        circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)
        swapped = not swapped

    assert swapped is False

    return circuit
