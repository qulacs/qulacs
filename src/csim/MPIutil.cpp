//
#ifdef _USE_MPI
#include "MPIutil.hpp"

#include "utility.hpp"

#define _NQUBIT_WORK 22  // 4 Mi x 16 Byte(CTYPE)

static MPI_Comm mpicomm = 0;
static int mpirank = 0;
static int mpisize = 0;
static int mpitag = 0;
static MPIutil mpiutil = NULL;
static MPI_Status mpistat;
// static pthread_mutex_t mutex= PTHREAD_MUTEX_INITIALIZER;
static CTYPE *workarea = NULL;

#define _MAX_REQUESTS 4  // 2 (isend/irecv) * 2 (double buffering)
static MPI_Request mpireq[_MAX_REQUESTS];
static UINT mpireq_idx = 0;
static UINT mpireq_cnt = 0;

static MPI_Request *get_request() {
    if (mpireq_cnt >= _MAX_REQUESTS) {
        fprintf(stderr, "cannot get a request for communication, %s, %d\n",
            __FILE__, __LINE__);
        exit(1);
    }

    mpireq_cnt++;
    MPI_Request *ret = &(mpireq[mpireq_idx]);
    mpireq_idx = (mpireq_idx + 1) % _MAX_REQUESTS;
    return ret;
}

static void mpi_wait(UINT count) {
    if (mpireq_cnt < count) {
        fprintf(stderr,
            "mpi_wait count(=%d) is over incompleted requests(=%d), %s, %d\n",
            count, mpireq_cnt, __FILE__, __LINE__);
        exit(1);
    }

    for (UINT i = 0; i < count; i++) {
        UINT idx = (_MAX_REQUESTS + mpireq_idx - mpireq_cnt) % _MAX_REQUESTS;
        MPI_Wait(&(mpireq[idx]), &mpistat);
        mpireq_cnt--;
    }
}

static int get_rank() { return mpirank; }

static int get_size() { return mpisize; }

static int get_tag() {
    // pthread_mutex_lock(&mutex);
    mpitag ^= 1 << 20;  // max 1M-ranks
    // pthread_mutex_unlock(&mutex);
    return mpitag;
}

static void release_workarea() {
    if (workarea != NULL) free(workarea);
    workarea = NULL;
}

static CTYPE *get_workarea(ITYPE *dim_work, ITYPE *num_work) {
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
            fprintf(stderr, "Can't malloc for variable, %s, %d\n", __FILE__,
                __LINE__);
            exit(1);
        }
    }
    return workarea;
}

static void barrier() { MPI_Barrier(mpicomm); }

static void m_DC_allgather(void *sendbuf, void *recvbuf, int count) {
    MPI_Allgather(sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, recvbuf, count,
        MPI_CXX_DOUBLE_COMPLEX, mpicomm);
}

static void m_DC_send(void *sendbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    MPI_Send(sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, tag0, mpicomm);
}

static void m_DC_recv(void *recvbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    MPI_Recv(recvbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, tag0, mpicomm,
        &mpistat);
}

static void m_DC_sendrecv(
    void *sendbuf, void *recvbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    int mpi_tag1 = tag0 + ((mpirank & pair_rank) << 1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;
    MPI_Sendrecv(sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, mpi_tag1,
        recvbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, mpi_tag2, mpicomm,
        &mpistat);
}

static void m_DC_sendrecv_replace(void *buf, int count, int pair_rank) {
    int tag0 = get_tag();
    int mpi_tag1 = tag0 + ((mpirank & pair_rank) << 1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;
    MPI_Sendrecv_replace(buf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank,
        mpi_tag1, pair_rank, mpi_tag2, mpicomm, &mpistat);
}

static void m_DC_isendrecv(
    void *sendbuf, void *recvbuf, int count, int pair_rank) {
    int tag0 = get_tag();
    int mpi_tag1 = tag0 + ((mpirank & pair_rank) << 1) + (mpirank > pair_rank);
    int mpi_tag2 = mpi_tag1 ^ 1;
    MPI_Request *send_request = get_request();
    MPI_Request *recv_request = get_request();

    MPI_Isend(sendbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, mpi_tag1,
        mpicomm, send_request);
    MPI_Irecv(recvbuf, count, MPI_CXX_DOUBLE_COMPLEX, pair_rank, mpi_tag2,
        mpicomm, recv_request);
}

static void m_I_allreduce(void *buf, UINT count) {
    MPI_Allreduce(
        MPI_IN_PLACE, buf, count, MPI_LONG_LONG_INT, MPI_SUM, mpicomm);
}

static void s_D_allgather(double a, void *recvbuf) {
    MPI_Allgather(&a, 1, MPI_DOUBLE, recvbuf, 1, MPI_DOUBLE, mpicomm);
}

static void s_D_allreduce(void *buf) {
    MPI_Allreduce(MPI_IN_PLACE, buf, 1, MPI_DOUBLE, MPI_SUM, mpicomm);
}

static void s_D_allreduce_ordersafe(void *buf) {
    double *recvbuf = (double *)malloc(mpisize * sizeof(double));
    MPI_Allgather(buf, 1, MPI_DOUBLE, recvbuf, 1, MPI_DOUBLE, mpicomm);
    double *sum = (double *)buf;
    *sum = 0.;
    for (int idx = 0; idx < mpisize; ++idx) {
        *sum += recvbuf[idx];
    }
    free(recvbuf);
}

static void s_u_bcast(UINT *a) { MPI_Bcast(a, 1, MPI_INT, 0, mpicomm); }

static void s_D_bcast(double *a) { MPI_Bcast(a, 1, MPI_DOUBLE, 0, mpicomm); }

#define REGISTER_METHOD_POINTER(M) mpiutil->M = M;

MPIutil get_mpiutil() {
    if (mpiutil != NULL) {
        return mpiutil;
    }

    mpicomm = MPI_COMM_WORLD;
    // printf("# MPI_COMM_WORLD %p\n", mpicomm);
    MPI_Comm_rank(mpicomm, &mpirank);
    MPI_Comm_size(mpicomm, &mpisize);
    mpitag = 0;

    mpiutil = (MPIutil)malloc(sizeof(*mpiutil));
    REGISTER_METHOD_POINTER(get_rank)
    REGISTER_METHOD_POINTER(get_size)
    REGISTER_METHOD_POINTER(get_tag)
    REGISTER_METHOD_POINTER(get_workarea)
    REGISTER_METHOD_POINTER(release_workarea)
    REGISTER_METHOD_POINTER(barrier)
    REGISTER_METHOD_POINTER(mpi_wait)
    REGISTER_METHOD_POINTER(m_DC_allgather)
    REGISTER_METHOD_POINTER(m_DC_send)
    REGISTER_METHOD_POINTER(m_DC_recv)
    REGISTER_METHOD_POINTER(m_DC_sendrecv)
    REGISTER_METHOD_POINTER(m_DC_sendrecv_replace)
    REGISTER_METHOD_POINTER(m_DC_isendrecv)
    REGISTER_METHOD_POINTER(m_I_allreduce)
    REGISTER_METHOD_POINTER(s_D_allgather)
    REGISTER_METHOD_POINTER(s_D_allreduce)
    REGISTER_METHOD_POINTER(s_D_allreduce_ordersafe)
    REGISTER_METHOD_POINTER(s_u_bcast)
    REGISTER_METHOD_POINTER(s_D_bcast)
    return mpiutil;
}

#undef REGISTER_METHOD_POINTER
#endif  // #ifdef _USE_MPI
