"""Regression tests for chunked MPI Pauli expectation values.

Run in an MPI-enabled Qulacs environment from the repository root:
    mpirun -n 2 python -m pytest -v \
        python/tests/multi_cpu/test_expectation_chunk_multi_cpu.py

MPIutil.hpp caps the receive buffer at 2**22 amplitudes. With two ranks,
23 local qubits require two communication chunks; 22 local qubits provide
the single-chunk baseline. Each rank needs up to 128 MiB for the state and
64 MiB for the receive buffer, plus runtime overhead. No serial reference
state or full-vector gather is needed.
"""

import numpy as np
import pytest

import qulacs


@pytest.fixture(scope="module", params=[22, 23], ids=["single_chunk", "two_chunks"])
def distributed_state(request, init_mpi):
    multicpu, mpicomm = init_mpi
    if not multicpu:
        pytest.skip("Requires MPI-enabled Qulacs and mpi4py")
    if mpicomm.Get_size() != 2:
        pytest.skip("Run with exactly two MPI ranks")

    # Keep this boundary in sync with _NQUBIT_WORK in src/csim/MPIutil.hpp.
    local_qubits = request.param
    state = qulacs.QuantumState(local_qubits + 1, use_multi_cpu=True)
    assert state.get_device_name() == "multi-cpu"
    return state, local_qubits


def assert_expectation(state, pauli_string, expected):
    observable = qulacs.Observable(state.get_qubit_count())
    observable.add_operator(1.0, pauli_string)
    # Every rank must call the expectation routine, including its MPI reduction.
    actual = observable.get_expectation_value(state)
    np.testing.assert_allclose(
        actual, expected, atol=1e-12, rtol=0, err_msg=pauli_string
    )


def test_global_x_expectation(distributed_state):
    state, local_qubits = distributed_state
    state.set_computational_basis(0)
    circuit = qulacs.QuantumCircuit(state.get_qubit_count())
    for target in range(state.get_qubit_count()):
        circuit.add_H_gate(target)
    circuit.update_quantum_state(state)

    # On |+>**n, X on the global qubit has expectation 1. For two chunks,
    # indexing recvptr by the full local index reads past its allocation.
    assert_expectation(state, f"X {local_qubits}", 1.0)


@pytest.mark.parametrize("pauli,expected", [("X", -1.0), ("Y", 1.0), ("Z", 1.0)])
def test_phi_minus_expectation(distributed_state, pauli, expected):
    state, local_qubits = distributed_state
    target = local_qubits - 1
    global_target = local_qubits
    state.set_computational_basis(0)

    # Prepare (|00> - |11>)/sqrt(2) on the highest local/global pair and |+>
    # on every other local qubit. With 23 local qubits, rank 1's nonzero
    # amplitudes fill the second chunk, while rank 0's fill the first.
    # Both XX and YY therefore require the entire second received chunk
    # AND a full local index into state. Filling the chunk also avoids
    # relying on a single out-of-bounds read returning an incorrect value.
    circuit = qulacs.QuantumCircuit(state.get_qubit_count())
    for spectator in range(target):
        circuit.add_H_gate(spectator)
    circuit.add_X_gate(target)
    circuit.add_H_gate(target)
    circuit.add_CNOT_gate(target, global_target)
    circuit.update_quantum_state(state)

    # Separate pytest cases ensure an XX assertion failure does not hide YY.
    assert_expectation(state, f"{pauli} {target} {pauli} {global_target}", expected)
