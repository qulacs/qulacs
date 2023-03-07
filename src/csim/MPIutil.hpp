//
//#ifdef _USE_MPI
#pragma once
#define OMPI_SKIP_MPICXX \
    1  // Fixes compilation with GCC8+
       // https://github.com/open-mpi/ompi/issues/5157

#ifdef _USE_MPI
#include <mpi.h>

#include <cassert>

#include "cppsim/exception.hpp"
#include "type.hpp"

#define _NQUBIT_WORK 22  // 4 Mi x 16 Byte(CTYPE)
#define _MAX_REQUESTS 4  // 2 (isend/irecv) * 2 (double buffering)

class MPIutil {
private:
    MPI_Comm mpicomm = 0;
    int mpirank = 0;
    int mpisize = 0;
    int mpitag = 0;
    MPI_Status mpistat;
    CTYPE *workarea = NULL;
    MPI_Request mpireq[_MAX_REQUESTS];
    UINT mpireq_idx = 0;
    UINT mpireq_cnt = 0;

    static void MPIFunctionError(
        const std::string &func, UINT ret, const std::string &file, UINT line);

    MPIutil() {
        mpicomm = MPI_COMM_WORLD;
        UINT ret;
        ret = MPI_Comm_rank(mpicomm, &mpirank);
        if (ret != MPI_SUCCESS)
            MPIFunctionError("MPI_Comm_rank", ret, __FILE__, __LINE__);
        ret = MPI_Comm_size(mpicomm, &mpisize);
        if (ret != MPI_SUCCESS)
            MPIFunctionError("MPI_Comm_size", ret, __FILE__, __LINE__);
        mpitag = 0;
    }
    ~MPIutil() = default;

public:
    MPIutil(const MPIutil &) = delete;
    MPIutil &operator=(const MPIutil &) = delete;
    MPIutil(MPIutil &&) = delete;
    MPIutil &operator=(MPIutil &&) = delete;

    static MPIutil &get_inst() {
        static MPIutil instance;
        return instance;
    }

    MPI_Request *get_request();
    int get_rank();
    int get_size();
    int get_tag();
    CTYPE *get_workarea(ITYPE *dim_work, ITYPE *num_work);
    void release_workarea();
    void barrier();
    void mpi_wait(UINT count);
    void m_DC_allgather(void *sendbuf, void *recvbuf, int count);
    void m_DC_send(void *sendbuf, int count, int pair_rank);
    void m_DC_recv(void *recvbuf, int count, int pair_rank);
    void m_DC_sendrecv(void *sendbuf, void *recvbuf, int count, int pair_rank);
    void m_DC_sendrecv_replace(void *buf, int count, int pair_rank);
    void m_DC_isendrecv(void *sendbuf, void *recvbuf, int count, int pair_rank);
    void m_I_allreduce(void *buf, UINT count);
    void s_D_allgather(double a, void *recvbuf);
    void s_D_allreduce(void *buf);
    void s_DC_allreduce(void *buf);
    void s_u_bcast(UINT *a);
    void s_D_bcast(double *a);
};
#endif
