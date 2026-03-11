"""Tests for MPI-aware PauliGate and PauliRotationGate.

Each test applies the same gate to a serial QuantumState and to a distributed
QuantumState(n, use_multi_cpu=True), then verifies that each rank's local
amplitude slice matches the corresponding slice of the serial reference.

Tests cover all three internal cases of the MPI implementation:

  Case A-Z   -- pure Z/I Pauli string; no communication required.
  Case A-XZ  -- local X/Y + global Z/I; no communication required.
  Case B1    -- all X/Y qubits are global; sendrecv + element-wise update.
  Case B2    -- mixed local and global X/Y; sendrecv + pairwise update.

Run with:
    mpirun -n 2 pytest python/tests/multi_cpu/test_gate_pauli_multi_cpu.py
    mpirun -n 4 pytest python/tests/multi_cpu/test_gate_pauli_multi_cpu.py
"""
from typing import Generator

import numpy as np
import pytest

import qulacs
import qulacs.gate as g

pytestmark = pytest.mark.skipif(
    not qulacs.check_build_for_mpi(),
    reason="To use multi-cpu, qulacs built for mpi is required.",
)

# Fixed problem size: n=6 gives inner_qc>=4 for up to 4 ranks.
_N = 6
_THETA = 0.7

# Local qubit indices (bits 0-3; always local for <=4 ranks with n=6)
_Q0, _Q1, _Q2 = 0, 1, 2

# Global qubit indices: highest bits carry rank information.
# _G0 (qubit 5) is global for all multi-rank runs.
# _G1 (qubit 4) is global only when outer_qc >= 2 (i.e., 4+ ranks);
# it is local with 2 ranks but the test is still correct in that case.
_G0 = _N - 1  # 5
_G1 = _N - 2  # 4

# == test-case tables ==========================================================
# Each entry: (label, basis, qubit_list, pauli_id_list)

_PAULI_GATE_CASES = [
    # Case A-XZ: local-only (all qubits < inner_qc)
    ("X_local[q0]|0001>", 0b000001, [_Q0], [1]),
    ("Y_local[q0]|0001>", 0b000001, [_Q0], [2]),
    ("Z_local[q0]|0001>", 0b000001, [_Q0], [3]),
    ("XY_local|0011>", 0b000011, [_Q0, _Q1], [1, 2]),
    ("XYZ_local|0111>", 0b000111, [_Q0, _Q1, _Q2], [1, 2, 3]),
    # Case A-Z: global Z/I only (bit_flip_mask==0, no communication)
    ("Z_global[g0]|000001>", 0b000001, [_G0], [3]),
    ("Z_global[g0]|100001>", 1 << _G0 | 1, [_G0], [3]),
    ("ZZ_local_global|000001>", 0b000001, [_Q0, _G0], [3, 3]),
    ("ZZ_local_global|000000>", 0b000000, [_Q0, _G0], [3, 3]),
    ("ZZ_local_global|100001>", 1 << _G0 | 1, [_Q0, _G0], [3, 3]),
    # Case A-XZ: local X/Y + global Z (no communication)
    ("X_local_Z_global|000001>", 0b000001, [_Q0, _G0], [1, 3]),
    ("Y_local_Z_global|000001>", 0b000001, [_Q0, _G0], [2, 3]),
    ("XY_local_Z_global|000011>", 0b000011, [_Q0, _Q1, _G0], [1, 2, 3]),
    ("X_local_Z_global|100001>", 1 << _G0 | 1, [_Q0, _G0], [1, 3]),
    # Case B1: global X/Y only (sendrecv + element-wise swap)
    ("X_global[g0]|000000>", 0b000000, [_G0], [1]),
    ("X_global[g0]|000001>", 0b000001, [_G0], [1]),
    ("Y_global[g0]|000000>", 0b000000, [_G0], [2]),
    ("Y_global[g0]|000001>", 0b000001, [_G0], [2]),
    ("Z_local_X_global|000001>", 0b000001, [_Q0, _G0], [3, 1]),
    ("Z_local_Y_global|000001>", 0b000001, [_Q0, _G0], [3, 2]),
    ("Y_local_X_global|000001>", 0b000001, [_Q1, _G0], [2, 1]),
    # Case B2: mixed local + global X/Y (sendrecv + pairwise swap)
    ("XX_local_global|000001>", 0b000001, [_Q0, _G0], [1, 1]),
    ("YY_local_global|000001>", 0b000001, [_Q0, _G0], [2, 2]),
    ("XY_local_global|000001>", 0b000001, [_Q0, _G0], [1, 2]),
    ("YX_local_global|000001>", 0b000001, [_Q0, _G0], [2, 1]),
    ("YZ_local_global|000001>", 0b000001, [_Q0, _G0], [2, 3]),
    ("XX_local_global|100001>", 1 << _G0 | 1, [_Q0, _G0], [1, 1]),
    ("YY_local_global|100000>", 1 << _G0, [_Q0, _G0], [2, 2]),
    ("ZY_local_global|000110>", 0b000110, [_Q1, _G0], [3, 2]),
    # Multi-global (g1=qubit4 is global for 4+ ranks, local for 2 ranks)
    ("XX_two_global|000000>", 0b000000, [_G0, _G1], [1, 1]),
    ("YY_two_global|000000>", 0b000000, [_G0, _G1], [2, 2]),
    ("XY_two_global|000001>", 0b000001, [_G0, _G1], [1, 2]),
    ("ZZ_two_global|000000>", 0b000000, [_G0, _G1], [3, 3]),
    ("X_local_XX_global|000001>", 0b000001, [_Q0, _G0, _G1], [1, 1, 1]),
]

# Each entry: (label, basis, qubit_list, pauli_id_list, angle)
_PAULI_ROT_CASES = [
    # Case A-XZ: local-only
    ("X_rot_local[q0]|000001>", 0b000001, [_Q0], [1], _THETA),
    ("Y_rot_local[q0]|000001>", 0b000001, [_Q0], [2], _THETA),
    ("Z_rot_local[q0]|000001>", 0b000001, [_Q0], [3], _THETA),
    ("XY_rot_local|000011>", 0b000011, [_Q0, _Q1], [1, 2], _THETA),
    # Case A-Z: global Z/I only
    ("Z_rot_global[g0]|000001>", 0b000001, [_G0], [3], _THETA),
    ("Z_rot_global[g0]|100001>", 1 << _G0 | 1, [_G0], [3], _THETA),
    ("ZZ_rot_local_global|000001>", 0b000001, [_Q0, _G0], [3, 3], _THETA),
    ("ZZ_rot_local_global|100001>", 1 << _G0 | 1, [_Q0, _G0], [3, 3], _THETA),
    ("ZZ_rot_local_global|000000>", 0b000000, [_Q0, _G0], [3, 3], _THETA),
    # Case A-XZ: local X/Y + global Z
    ("X_rot_local_Z_global|000001>", 0b000001, [_Q0, _G0], [1, 3], _THETA),
    ("Y_rot_local_Z_global|000001>", 0b000001, [_Q0, _G0], [2, 3], _THETA),
    ("X_rot_local_Z_global|100001>", 1 << _G0 | 1, [_Q0, _G0], [1, 3], _THETA),
    # Case B1: global X/Y only
    ("X_rot_global[g0]|000001>", 0b000001, [_G0], [1], _THETA),
    ("Y_rot_global[g0]|000001>", 0b000001, [_G0], [2], _THETA),
    ("Z_rot_local_X_global|000001>", 0b000001, [_Q0, _G0], [3, 1], _THETA),
    ("Z_rot_local_Y_global|000001>", 0b000001, [_Q0, _G0], [3, 2], _THETA),
    # Case B2: mixed local + global X/Y
    ("XX_rot|000011>", 0b000011, [_Q0, _G0], [1, 1], 1.0),
    ("XX_rot|000001>", 0b000001, [_Q0, _G0], [1, 1], _THETA),
    ("YY_rot|000001>", 0b000001, [_Q0, _G0], [2, 2], _THETA),
    ("XY_rot|000001>", 0b000001, [_Q0, _G0], [1, 2], _THETA),
    ("YX_rot|000001>", 0b000001, [_Q0, _G0], [2, 1], _THETA),
    ("YZ_rot|000001>", 0b000001, [_Q0, _G0], [2, 3], _THETA),
    ("XX_rot|100001>", 1 << _G0 | 1, [_Q0, _G0], [1, 1], _THETA),
    ("YY_rot|100000>", 1 << _G0, [_Q0, _G0], [2, 2], _THETA),
    # Multi-global (g1 is global for 4+ ranks)
    ("XX_rot_two_global|000001>", 0b000001, [_G0, _G1], [1, 1], _THETA),
    ("YY_rot_two_global|000001>", 0b000001, [_G0, _G1], [2, 2], _THETA),
    ("ZZ_rot_two_global|000001>", 0b000001, [_G0, _G1], [3, 3], _THETA),
    ("XXX_rot_local_2global|000001>", 0b000001, [_Q0, _G0, _G1], [1, 1, 1], _THETA),
    ("YZX_rot_mixed|000011>", 0b000011, [_Q1, _G0, _G1], [2, 3, 1], _THETA),
    # Full 3x3 sweep of all (local, global) Pauli pairs
    ("XX_rot_sweep", 0b000001, [_Q0, _G0], [1, 1], _THETA),
    ("XY_rot_sweep", 0b000001, [_Q0, _G0], [1, 2], _THETA),
    ("XZ_rot_sweep", 0b000001, [_Q0, _G0], [1, 3], _THETA),
    ("YX_rot_sweep", 0b000001, [_Q0, _G0], [2, 1], _THETA),
    ("YY_rot_sweep", 0b000001, [_Q0, _G0], [2, 2], _THETA),
    ("YZ_rot_sweep", 0b000001, [_Q0, _G0], [2, 3], _THETA),
    ("ZX_rot_sweep", 0b000001, [_Q0, _G0], [3, 1], _THETA),
    ("ZY_rot_sweep", 0b000001, [_Q0, _G0], [3, 2], _THETA),
    ("ZZ_rot_sweep", 0b000001, [_Q0, _G0], [3, 3], _THETA),
]


# == test classes ==============================================================


class TestPauliGateMultiCpu:
    @pytest.fixture
    def init_state(self, init_mpi) -> Generator[None, None, None]:
        multicpu, mpicomm = init_mpi
        if multicpu:
            self.mpirank = mpicomm.Get_rank()  # type: ignore
            self.mpisize = mpicomm.Get_size()  # type: ignore
        else:
            self.mpirank = 0
            self.mpisize = 1
        self.dim_local = 2**_N // self.mpisize
        yield

    def _check(self, gate, basis) -> None:
        """Apply gate; assert MPI local slice == serial reference slice."""
        ref = qulacs.QuantumState(_N)
        ref.set_computational_basis(basis)
        gate.update_quantum_state(ref)
        ref_vec = np.array(ref.get_vector())
        local_ref = ref_vec[
            self.dim_local * self.mpirank : self.dim_local * (self.mpirank + 1)
        ]

        mpi_s = qulacs.QuantumState(_N, True)
        mpi_s.set_computational_basis(basis)
        gate.update_quantum_state(mpi_s)

        np.testing.assert_allclose(np.array(mpi_s.get_vector()), local_ref, atol=1e-10)

    @pytest.mark.parametrize("label,basis,qubits,pauli_ids", _PAULI_GATE_CASES)
    @pytest.mark.usefixtures("init_state")
    def test_pauli_gate(self, label, basis, qubits, pauli_ids) -> None:
        self._check(g.Pauli(qubits, pauli_ids), basis)

    @pytest.mark.parametrize("label,basis,qubits,pauli_ids,angle", _PAULI_ROT_CASES)
    @pytest.mark.usefixtures("init_state")
    def test_pauli_rotation_gate(self, label, basis, qubits, pauli_ids, angle) -> None:
        self._check(g.PauliRotation(qubits, pauli_ids, angle), basis)
