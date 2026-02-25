//
#ifdef _USE_MPI
#include "MPIutil.hpp"

#include "utility.hpp"

void MPIutil::MPIFunctionError(
    const std::string &func, UINT ret, const std::string &file, UINT line) {
    std::string msg1 = func;
    std::string msg2 = " error(=";
    std::string msg3 = std::to_string(ret);
    std::string msg4 = "), ";
    std::string msg5 = file;
    std::string msg6 = ", ";
    std::string msg7 = std::to_string(line);
    throw MPIRuntimeException(msg1 + msg2 + msg3 + msg4 + msg5 + msg6 + msg7);
}

MPI_Request *MPIutil::get_request() {
    if (mpireq_cnt >= _MAX_REQUESTS) {
        std::string msg1 = "cannot get a request for communication, ";
        std::string msg2 = __FILE__;
        std::string msg3 = ", ";
        std::string msg4 = std::to_string(__LINE__);
        throw MPIRuntimeException(msg1 + msg2 + msg3 + msg4);
    }

    mpireq_cnt++;
    MPI_Request *ret = &(mpireq[mpireq_idx]);
    mpireq_idx = (mpireq_idx + 1) % _MAX_REQUESTS;
    return ret;
}

void MPIutil::mpi_wait(UINT count) {
    if (mpireq_cnt < count) {
        std::string msg1 = "mpi_wait count(=";
        std::string msg2 = std::to_string(count);
        std::string msg3 = ") is over incompleted requests(=";
        std::string msg4 = std::to_string(mpireq_cnt);
        std::string msg5 = "), ";
        std::string msg6 = __FILE__;
        std::string msg7 = ", ";
        std::string msg8 = std::to_string(__LINE__);
        throw MPIRuntimeException(
            msg1 + msg2 + msg3 + msg4 + msg5 + msg6 + msg7 + msg8);
    }

    for (UINT i = 0; i < count; i++) {
        UINT idx = (_MAX_REQUESTS + mpireq_idx - mpireq_cnt) % _MAX_REQUESTS;
        UINT ret = MPI_Wait(&(mpireq[idx]), &mpistat);
        if (ret != MPI_SUCCESS)
            MPIFunctionError("MPI_Wait", ret, __FILE__, __LINE__);
        mpireq_cnt--;
    }
}

int MPIutil::get_rank() { return mpirank; }

int MPIutil::get_size() { return mpisize; }

int MPIutil::get_tag() {
    // pthread_mutex_lock(&mutex);
    mpitag ^= 1 << 20;  // max 1M-ranks
    // pthread_mutex_unlock(&mutex);
    return mpitag;
}

void MPIutil::release_workarea() {
    if (workarea != NULL) free(workarea);
    workarea = NULL;
}

CTYPE *MPIutil::get_workarea(ITYPE *dim_work, ITYPE *num_work) {
    ITYPE dim = *dim_work;
    *dim_work = get_min_ll(1 << _NQUBIT_WORK, dim);
    *num_work = get_max_ll(1, dim >> _NQUBIT_WORK);
    if (workarea == NULL) {
#if defined(__ARM_FEATURE_SVE)
        posix_memalign(
            (void **)&workarea, 256, sizeof(CTYPE) * (1 << _NQUBIT_WORK));
#else
        workarea = (CTYPE *)malloc(sizeof(CTYPE) * (1 << _NQUBIT_WORK));
#endif
        if (workarea == NULL) {
            std::string msg1 = "Can't malloc in get_workarea for MPI, ";
            std::string msg2 = __FILE__;
            std::string msg3 = ", ";
            std::string msg4 = std::to_string(__LINE__);
            throw MPIRuntimeException(msg1 + msg2 + msg3 + msg4);
        }
    }
    return workarea;
}

void MPIutil::barrier() {
    UINT ret = MPI_Barrier(mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Barrier", ret, __FILE__, __LINE__);
}

void MPIutil::m_DC_send(void *sendbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    UINT ret = MPI_Send(
        sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, tag0, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Send", ret, __FILE__, __LINE__);
}

void MPIutil::m_DC_recv(void *recvbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    UINT ret = MPI_Recv(recvbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, tag0,
        mpicomm, &mpistat);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Recv", ret, __FILE__, __LINE__);
}

void MPIutil::m_DC_sendrecv(
    void *sendbuf, void *recvbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    int mpi_tag1 = tag0 + ((mpirank & pair_rank) << 1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;
    UINT ret = MPI_Sendrecv(sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank,
        mpi_tag1, recvbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, mpi_tag2,
        mpicomm, &mpistat);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Sendrecv", ret, __FILE__, __LINE__);
}

void MPIutil::m_DC_sendrecv_replace(void *buf, int count, int pair_rank) {
    int tag0 = get_tag();
    int mpi_tag1 = tag0 + ((mpirank & pair_rank) << 1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;
    UINT ret = MPI_Sendrecv_replace(buf, count, MPI_CXX_DOUBLE_COMPLEX,
        pair_rank, mpi_tag1, pair_rank, mpi_tag2, mpicomm, &mpistat);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Sendrecv_replace", ret, __FILE__, __LINE__);
}

void MPIutil::m_DC_isendrecv(
    void *sendbuf, void *recvbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    int mpi_tag1 = tag0 + ((mpirank & pair_rank) << 1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;
    MPI_Request *send_request = get_request();
    MPI_Request *recv_request = get_request();

    UINT ret = MPI_Isend(sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank,
        mpi_tag1, mpicomm, send_request);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Isend", ret, __FILE__, __LINE__);
    ret = MPI_Irecv(recvbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, mpi_tag2,
        mpicomm, recv_request);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Irecv", ret, __FILE__, __LINE__);
}

void MPIutil::m_DC_allgather(void *sendbuf, void *recvbuf, int count) {
    UINT ret = MPI_Allgather(sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, recvbuf,
        count, MPI_CXX_DOUBLE_COMPLEX, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Allgather<CTYPE>", ret, __FILE__, __LINE__);
}

void MPIutil::s_D_allgather(double a, void *recvbuf) {
    UINT ret =
        MPI_Allgather(&a, 1, MPI_DOUBLE, recvbuf, 1, MPI_DOUBLE, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Allgather<DOUBLE>", ret, __FILE__, __LINE__);
}

void MPIutil::m_I_allreduce(void *buf, UINT count) {
    UINT ret = MPI_Allreduce(
        MPI_IN_PLACE, buf, count, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Allreduce<ITYPE>", ret, __FILE__, __LINE__);
}

void MPIutil::s_D_allreduce(void *buf) {
    UINT ret =
        MPI_Allreduce(MPI_IN_PLACE, buf, 1, MPI_DOUBLE, MPI_SUM, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Allreduce<DOUBLE>", ret, __FILE__, __LINE__);
}

void MPIutil::s_DC_allreduce(void *buf) {
    UINT ret = MPI_Allreduce(
        MPI_IN_PLACE, buf, 1, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Allreduce<CTYPE>", ret, __FILE__, __LINE__);
}

void MPIutil::s_u_bcast(UINT *a) {
    UINT ret = MPI_Bcast(a, 1, MPI_INT, 0, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Bcast<int>", ret, __FILE__, __LINE__);
}

void MPIutil::s_D_bcast(double *a) {
    UINT ret = MPI_Bcast(a, 1, MPI_DOUBLE, 0, mpicomm);
    if (ret != MPI_SUCCESS)
        MPIFunctionError("MPI_Bcast<DOUBLE>", ret, __FILE__, __LINE__);
}
#endif  // #ifdef _USE_MPI
