"""
build a QuantumVolume Circuit with parameter
"""

import numpy as np

from qulacs import QuantumCircuit


def simple_swap(p, q, array):
    tmp = array[p]
    array[p] = array[q]
    array[q] = tmp


def local_swap(p, q, done_ug, qubit_table):
    simple_swap(p, q, done_ug)
    simple_swap(p, q, qubit_table)
    # simple_swap(p, q, master_table)


def block_swap(p, q, bs, done_ug, qubit_table):
    for t in range(bs):
        simple_swap(p + t, q + t, qubit_table)
        # simple_swap(p + t, q + t, master_table)


def build_circuit(nqubits, global_nqubits=0, depth=10, verbose=False, random_gen=""):
    use_fusedswap = True if global_nqubits > 0 else False
    local_nqubits = nqubits - global_nqubits
    if random_gen == "":
        rng = np.random.default_rng()
    else:
        rng = random_gen

    circuit = QuantumCircuit(nqubits)
    perm_0 = list(range(nqubits))
    seed = rng.integers(10000)

    for d in range(depth):
        qubit_table = list(range(nqubits))
        perm = rng.permutation(perm_0)
        pend_pair = []
        done_ug = [0] * nqubits

        # add random_unitary_gate for local_nqubits, first
        for w in range(nqubits // 2):
            physical_qubits = [int(perm[2 * w]), int(perm[2 * w + 1])]
            if (
                physical_qubits[0] < local_nqubits
                and physical_qubits[1] < local_nqubits
            ):
                if verbose:
                    print("#1: circuit.add_random_unitary_gate(", physical_qubits, ")")
                circuit.add_random_unitary_gate(physical_qubits, seed)
                seed += 1
                done_ug[physical_qubits[0]] = 1
                done_ug[physical_qubits[1]] = 1
            else:
                pend_pair.append(physical_qubits)

        # add SWAP gate for FusedSWAP
        work_qubit = local_nqubits - global_nqubits
        for s in range(global_nqubits):
            if done_ug[work_qubit + s] == 0:
                for t in range(work_qubit):
                    if done_ug[work_qubit - t - 1] == 1:
                        p = work_qubit + s
                        q = work_qubit - t - 1
                        local_swap(p, q, done_ug, qubit_table)
                        if verbose:
                            print("#2: circuit.add_SWAP_gate(", p, ", ", q, ")")
                        circuit.add_SWAP_gate(p, q)
                        break

        if verbose:
            print(
                "#3 block_swap(",
                work_qubit,
                ", ",
                local_nqubits,
                ", ",
                global_nqubits,
                ")",
            )
        if global_nqubits > 0:
            block_swap(work_qubit, local_nqubits, global_nqubits, done_ug, qubit_table)
            circuit.add_FusedSWAP_gate(work_qubit, local_nqubits, global_nqubits)
        if verbose:
            print("#: qubit_table=", qubit_table)

        # add random_unitary_gate for qubits that were originally outside.
        for pair in pend_pair:
            unitary_pair = [qubit_table.index(pair[0]), qubit_table.index(pair[1])]
            if verbose:
                print("#4: circuit.add_random_unitary_gate(", unitary_pair, ")")
            circuit.add_random_unitary_gate(unitary_pair, seed)
            seed += 1
            done_ug[unitary_pair[0]] = 1
            done_ug[unitary_pair[1]] = 1

    if verbose:
        print("circuit=", circuit)

    return circuit
