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


# == Regression tests: local qubit index >= _NQUBIT_WORK =======================
#
# _NQUBIT_WORK = 22 (MPIutil.hpp) caps the MPI work buffer at 2^22 amplitudes.
# When a local X/Y qubit has index >= 22, its partner element falls in a
# different buffer chunk (lbfm_high != 0).  Two bugs existed before the fix:
#
#   Bug 1 (OOB read):    j ^ local_bit_flip_mask >= dim_work → SIGSEGV
#   Bug 2 (wrong phase): phase computed from within-chunk index j instead of
#                        the full local index (iter * dim_work + j)
#
# Triggering either bug requires inner_qc > _NQUBIT_WORK.  With 2 MPI ranks
# and N=24: inner_qc = 23 > 22; each rank holds 2^23 = 8Mi amplitudes (~128 MB).
# Tests are skipped automatically when this condition is not met (e.g. 4 ranks
# with N=24 gives inner_qc=22, or when running without MPI).

_N_LARGE = 24
_NQUBIT_WORK = 22  # must match _NQUBIT_WORK in src/csim/MPIutil.hpp

# With 2 MPI ranks and N=24: qubit 22 is the highest local qubit (index 22)
# and qubit 23 is the single global qubit.
_QL22 = 22
_GL23 = _N_LARGE - 1  # = 23; global for 2 ranks

# Each tuple: (label, basis, qubit_list, pauli_id_list)
_LARGE_PAULI_GATE_CASES = [
    # Bug 1: cross-chunk OOB — X/Y on local qubit 22 (lbfm_high != 0)
    ("cross_chunk_XX|1>", 0b1, [_QL22, _GL23], [1, 1]),
    ("cross_chunk_XX|q22>", 1 << _QL22, [_QL22, _GL23], [1, 1]),
    ("cross_chunk_XY|1>", 0b1, [_QL22, _GL23], [1, 2]),
    # Bug 1 + 2: Y on local 22 contributes to local_phase_flip_mask too
    ("cross_chunk_YX|1>", 0b1, [_QL22, _GL23], [2, 1]),
    ("cross_chunk_YX|q22>", 1 << _QL22, [_QL22, _GL23], [2, 1]),
    # Bug 2: same-chunk B1 with iter>0 — Z on local 22 sets high phase-mask bit
    ("same_chunk_B1_ZX|1>", 0b1, [_QL22, _GL23], [3, 1]),
    ("same_chunk_B1_ZX|q22>", 1 << _QL22, [_QL22, _GL23], [3, 1]),
    ("same_chunk_B1_ZY|1>", 0b1, [_QL22, _GL23], [3, 2]),
]

# Each tuple: (label, basis, qubit_list, pauli_id_list, angle)
_LARGE_PAULI_ROT_CASES = [
    # Bug 1: cross-chunk OOB
    ("cross_chunk_rot_XX|1>", 0b1, [_QL22, _GL23], [1, 1], _THETA),
    ("cross_chunk_rot_XX|q22>", 1 << _QL22, [_QL22, _GL23], [1, 1], _THETA),
    ("cross_chunk_rot_XY|1>", 0b1, [_QL22, _GL23], [1, 2], _THETA),
    # Bug 1 + 2: Y on local 22
    ("cross_chunk_rot_YX|1>", 0b1, [_QL22, _GL23], [2, 1], _THETA),
    ("cross_chunk_rot_YX|q22>", 1 << _QL22, [_QL22, _GL23], [2, 1], _THETA),
    # Bug 2: same-chunk B1 with iter>0
    ("same_chunk_B1_rot_ZX|1>", 0b1, [_QL22, _GL23], [3, 1], _THETA),
    ("same_chunk_B1_rot_ZX|q22>", 1 << _QL22, [_QL22, _GL23], [3, 1], _THETA),
    ("same_chunk_B1_rot_ZY|1>", 0b1, [_QL22, _GL23], [3, 2], _THETA),
]


class TestPauliGateLargeChunk:
    """Regression tests for bugs triggered when local qubit index >= _NQUBIT_WORK.

    Requires 2 MPI ranks with N=24 so that inner_qc=23 > _NQUBIT_WORK=22.
    Tests are skipped automatically when that condition is not satisfied.
    """

    @pytest.fixture
    def init_state(self, init_mpi) -> Generator[None, None, None]:
        multicpu, mpicomm = init_mpi
        if multicpu:
            self.mpirank = mpicomm.Get_rank()  # type: ignore
            self.mpisize = mpicomm.Get_size()  # type: ignore
        else:
            self.mpirank = 0
            self.mpisize = 1
        outer_qc = (self.mpisize - 1).bit_length()  # log2(mpisize) for power-of-2
        inner_qc = _N_LARGE - outer_qc
        if inner_qc <= _NQUBIT_WORK:
            pytest.skip(
                f"inner_qc={inner_qc} <= _NQUBIT_WORK={_NQUBIT_WORK}: "
                f"need 2 MPI ranks with N={_N_LARGE} to exercise the cross-chunk path"
            )
        self.dim_local = 2**_N_LARGE // self.mpisize
        yield

    def _check(self, gate, basis) -> None:
        """Apply gate; assert MPI local slice == serial reference slice."""
        ref = qulacs.QuantumState(_N_LARGE)
        ref.set_computational_basis(basis)
        gate.update_quantum_state(ref)
        ref_vec = np.array(ref.get_vector())
        local_ref = ref_vec[
            self.dim_local * self.mpirank : self.dim_local * (self.mpirank + 1)
        ]

        mpi_s = qulacs.QuantumState(_N_LARGE, True)
        mpi_s.set_computational_basis(basis)
        gate.update_quantum_state(mpi_s)

        np.testing.assert_allclose(np.array(mpi_s.get_vector()), local_ref, atol=1e-10)

    @pytest.mark.parametrize("label,basis,qubits,pauli_ids", _LARGE_PAULI_GATE_CASES)
    @pytest.mark.usefixtures("init_state")
    def test_pauli_gate_large(self, label, basis, qubits, pauli_ids) -> None:
        self._check(g.Pauli(qubits, pauli_ids), basis)

    @pytest.mark.parametrize(
        "label,basis,qubits,pauli_ids,angle", _LARGE_PAULI_ROT_CASES
    )
    @pytest.mark.usefixtures("init_state")
    def test_pauli_rotation_gate_large(
        self, label, basis, qubits, pauli_ids, angle
    ) -> None:
        self._check(g.PauliRotation(qubits, pauli_ids, angle), basis)
