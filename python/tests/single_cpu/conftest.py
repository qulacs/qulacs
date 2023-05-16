import pytest

import qulacs


# Run once before all tests in the single_cpu directory
@pytest.fixture(scope="session", autouse=True)
def init_mpi() -> None:
    if qulacs.check_build_for_mpi():
        from mpi4py import MPI

        mpicomm = MPI.COMM_WORLD
        if mpicomm.Get_rank() == 0:
            print()
            print("Test with MPI. size=", mpicomm.Get_size())
