//
#ifdef _USE_MPI
#pragma once

#include <assert.h>
#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "type.hpp"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef struct {
    int (*get_rank)();
    int (*get_size)();
    int (*get_tag)();
    CTYPE *(*get_workarea)(ITYPE *dim_work, ITYPE *num_work);
    void (*release_workarea)();
    void (*barrier)();
    void (*mpi_wait)(UINT count);
    void (*m_DC_allgather)(void *sendbuf, void *recvbuf, int count);
    void (*m_DC_send)(void *sendbuf, int count, int pair_rank);
    void (*m_DC_recv)(void *recvbuf, int count, int pair_rank);
    void (*m_DC_sendrecv)(
        void *sendbuf, void *recvbuf, int count, int pair_rank);
    void (*m_DC_sendrecv_replace)(void *buf, int count, int pair_rank);
    void (*m_DC_isendrecv)(
        void *sendbuf, void *recvbuf, int count, int pair_rank);
    void (*m_I_allreduce)(void *buf, UINT count);
    void (*s_D_allgather)(double a, void *recvbuf);
    void (*s_D_allreduce)(void *buf);
    void (*s_D_allreduce_ordersafe)(void *buf);
    void (*s_u_bcast)(UINT *a);
    void (*s_D_bcast)(double *a);
} MPIutil_;
typedef MPIutil_ *MPIutil;

MPIutil get_mpiutil(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* _USE_MPI */
