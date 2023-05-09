from typing import Any, Tuple

import pytest

import qulacs


@pytest.fixture(scope="session")
def init_mpi() -> Tuple[bool, Any]:
    multicpu = False
    mpicomm = None

    if qulacs.check_build_for_mpi():
        try:
            from mpi4py import MPI

            mpicomm = MPI.COMM_WORLD
            if mpicomm.Get_rank() == 0:
                print("Test with MPI. size=", mpicomm.Get_size())

            multicpu = True
        except Exception as e:
            print(e)
            print("To use multi-cpu, the mpi4py library is required.")

    return multicpu, mpicomm
