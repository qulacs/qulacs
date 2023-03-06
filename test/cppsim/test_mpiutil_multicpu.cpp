#ifdef _USE_MPI
#include <gtest/gtest.h>

#include <cppsim/utility.hpp>
#include <csim/MPIutil.hpp>

#include "../util/util.hpp"

TEST(MPIutilTest_multicpu, SingletonCheck) {
    MPIutil& mpiutil_0 = MPIutil::get_inst();
    MPIutil& mpiutil_1 = MPIutil::get_inst();

    ASSERT_EQ(&mpiutil_0, &mpiutil_1);
}

TEST(MPIutilTest_multicpu, GetworkareaCheck) {
    // small size
    ITYPE dim_work = 1;
    ITYPE num_work = 0;
    MPIutil& m = MPIutil::get_inst();
    CTYPE* t = m.get_workarea(&dim_work, &num_work);

    ASSERT_EQ(dim_work, 1);
    ASSERT_EQ(num_work, 1);

    // large size
    dim_work = 1ULL << 40;
    num_work = 0;
    t = m.get_workarea(&dim_work, &num_work);

    ASSERT_EQ(dim_work * num_work, 1ULL << 40);
}

TEST(MPIutilTest_multicpu, SendRecvTest) {
    MPIutil& m = MPIutil::get_inst();
    const int mpirank = m.get_rank();
    const int mpisize = m.get_size();

    for (UINT pair_bit = 1; pair_bit < mpisize; ++pair_bit) {
        const UINT pair_rank = mpirank ^ pair_bit;
        CTYPE buf[2];
        CTYPE xdc[2] = {
            ((double)mpirank * 0.1) + ((double)(mpirank * mpirank) * 0.1i),
            ((double)mpirank * 0.3) + ((double)(mpirank * mpirank) * 0.7i)};
        CTYPE xdc_exp[2] = {((double)pair_rank * 0.1) +
                                ((double)(pair_rank * pair_rank) * 0.1i),
            ((double)pair_rank * 0.3) +
                ((double)(pair_rank * pair_rank) * 0.7i)};

        // send, recv
        if (mpirank < pair_rank) {  // send from smaller-rank
            m.m_DC_send(&xdc, 2, pair_rank);
            m.m_DC_recv(&buf, 2, pair_rank);
        } else {
            m.m_DC_recv(&buf, 2, pair_rank);
            m.m_DC_send(&xdc, 2, pair_rank);
        }
        ASSERT_NEAR(real(buf[0]), real(xdc_exp[0]), eps);
        ASSERT_NEAR(imag(buf[0]), imag(xdc_exp[0]), eps);
        ASSERT_NEAR(real(buf[1]), real(xdc_exp[1]), eps);
        ASSERT_NEAR(imag(buf[1]), imag(xdc_exp[1]), eps);

        // sendrecv
        buf[0] = 0.0 + 0.0i;
        buf[1] = 0.0 + 0.0i;
        m.m_DC_sendrecv(&xdc, &buf, 2, pair_rank);
        ASSERT_NEAR(real(buf[0]), real(xdc_exp[0]), eps);
        ASSERT_NEAR(imag(buf[0]), imag(xdc_exp[0]), eps);
        ASSERT_NEAR(real(buf[1]), real(xdc_exp[1]), eps);
        ASSERT_NEAR(imag(buf[1]), imag(xdc_exp[1]), eps);

        // isendrecv
        buf[0] = 0.0 + 0.0i;
        buf[1] = 0.0 + 0.0i;
        m.m_DC_isendrecv(&xdc, &buf, 2, pair_rank);
        m.mpi_wait(2);
        ASSERT_NEAR(real(buf[0]), real(xdc_exp[0]), eps);
        ASSERT_NEAR(imag(buf[0]), imag(xdc_exp[0]), eps);
        ASSERT_NEAR(real(buf[1]), real(xdc_exp[1]), eps);
        ASSERT_NEAR(imag(buf[1]), imag(xdc_exp[1]), eps);

        // sendrecv_replace
        buf[0] = 0.0 + 0.0i;
        buf[1] = 0.0 + 0.0i;
        m.m_DC_sendrecv_replace(&xdc, 2, pair_rank);
        ASSERT_NEAR(real(xdc[0]), real(xdc_exp[0]), eps);
        ASSERT_NEAR(imag(xdc[0]), imag(xdc_exp[0]), eps);
        ASSERT_NEAR(real(xdc[1]), real(xdc_exp[1]), eps);
        ASSERT_NEAR(imag(xdc[1]), imag(xdc_exp[1]), eps);

        m.barrier();  // no means..
    }
}

TEST(MPIutilTest_multicpu, BcastTest) {
    MPIutil& m = MPIutil::get_inst();
    int mpirank = m.get_rank();

    // single-UINT
    UINT xu = (UINT)mpirank * 2 + 1;  // rank0 has 1
    m.s_u_bcast(&xu);
    ASSERT_EQ(xu, 1);

    // single-double
    double xd = 1.1 * (double)mpirank + 0.1;  // rank0 has 0.1
    m.s_D_bcast(&xd);
    ASSERT_EQ(xd, 0.1);
}

TEST(MPIutilTest_multicpu, AllgatherTest) {
    MPIutil& m = MPIutil::get_inst();
    int mpirank = m.get_rank();
    int mpisize = m.get_size();

    ITYPE dim_work = mpisize * 2;
    ITYPE num_work = 0;
    CTYPE* t = m.get_workarea(&dim_work, &num_work);
    ASSERT_EQ(dim_work, mpisize * 2);
    ASSERT_EQ(num_work, 1);

    // single-double
    double* td = (double*)t;
    double xd = ((double)mpirank * 2) + 1.;
    m.s_D_allgather(xd, td);
    for (int iter = 0; iter < mpisize; ++iter, ++td) {
        ASSERT_EQ(*td, (double)iter * 2 + 1.);
    }

    // multi-CTYPE
    CTYPE xdc[2] = {
        ((double)mpirank * 2.1) + ((double)(mpirank * mpirank) * 1.0i),
        (0.5) + ((0.25 + mpirank) * 1.0i)};
    m.m_DC_allgather(xdc, t, 2);
    for (int iter = 0; iter < mpisize; ++iter, t += 2) {
        ASSERT_NEAR(real(t[0]), (double)iter * 2.1, eps);
        ASSERT_NEAR(imag(t[0]), (double)(iter * iter), eps);
        ASSERT_NEAR(real(t[1]), 0.5, eps);
        ASSERT_NEAR(imag(t[1]), 0.25 + iter, eps);
    }
}

TEST(MPIutilTest_multicpu, AllreduceTest) {
    MPIutil& m = MPIutil::get_inst();
    int mpirank = m.get_rank();
    int mpisize = m.get_size();

    // multi-ITYPE
    ITYPE xu[2] = {((ITYPE)mpirank << 32) + 1, (ITYPE)mpisize};
    m.m_I_allreduce(&xu, 2);
    ASSERT_EQ(xu[0], (((ITYPE)(mpisize - 1) * mpisize / 2) << 32) + mpisize);
    ASSERT_EQ(xu[1], (ITYPE)mpisize * mpisize);

    // single-double
    double xd = (double)mpirank * 1.1 + 0.1;
    m.s_D_allreduce(&xd);
    ASSERT_NEAR(xd,
        ((double)(mpisize - 1) * (double)mpisize / 2.) * 1.1 + 0.1 * mpisize,
        eps);

    // single-CTYPE
    CTYPE xdc = ((CTYPE)mpirank * 1.1) + 0.1i;
    m.s_DC_allreduce(&xdc);
    ASSERT_NEAR(
        real(xdc), ((double)(mpisize - 1) * (double)mpisize / 2.) * 1.1, eps);
    ASSERT_NEAR(imag(xdc), 0.1 * (double)mpisize, eps);
}
#endif
