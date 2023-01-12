#ifdef _OPENMP
#include <gtest/gtest.h>

#include <csim/utility.hpp>

TEST(OMPutilOperationTest, SingletonCheck) {
    OMPutil omputil_0 = get_omputil();
    OMPutil omputil_1 = get_omputil();

    ASSERT_EQ(omputil_0, omputil_1);
}

// Must be run under one of these conditions
// - QULACS_NUM_THREADS = OMP_NUM_THREADS
// - unset QULACS_NUM_THREADS
TEST(OMPutilOperationTest, SetResetNumThreadsCheck) {
    OMPutil omputil = get_omputil();

    UINT max_threads = omp_get_max_threads();

    omputil->set_qulacs_num_threads(1LL, 1);
    ASSERT_EQ(omp_get_max_threads(), 1);

    omputil->set_qulacs_num_threads(1LL << 10, 10);
    ASSERT_EQ(omp_get_max_threads(), max_threads);

    omputil->set_qulacs_num_threads(1LL << 9, 10);
    ASSERT_EQ(omp_get_max_threads(), 1);

    omputil->set_qulacs_num_threads(1LL << 34, 34);
    ASSERT_EQ(omp_get_max_threads(), max_threads);

    omputil->set_qulacs_num_threads(1LL << 33, 34);
    ASSERT_EQ(omp_get_max_threads(), 1);

    omputil->reset_qulacs_num_threads();
    ASSERT_EQ(omp_get_max_threads(), max_threads);
}
#endif
