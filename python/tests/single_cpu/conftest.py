import qulacs

if qulacs.check_build_for_mpi():
    from mpi4py import MPI

    mpicomm = MPI.COMM_WORLD
    if mpicomm.Get_rank() == 0:
        print("Test with MPI. size=", mpicomm.Get_size())
