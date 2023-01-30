#ifdef _OPENMP
#include <gtest/gtest.h>

#include <csim/utility.hpp>

TEST(OMPutilOperationTest, SingletonCheck) {
    OMPutil& omputil_0 = OMPutil::get_inst();
    OMPutil& omputil_1 = OMPutil::get_inst();

    ASSERT_EQ(&omputil_0, &omputil_1);
}

// Must be run under one of these conditions
// - QULACS_NUM_THREADS = OMP_NUM_THREADS
// - unset QULACS_NUM_THREADS
TEST(OMPutilOperationTest, SetResetNumThreadsCheck) {
    UINT max_threads = omp_get_max_threads();

    OMPutil::get_inst().set_qulacs_num_threads(1LL, 1);
    ASSERT_EQ(omp_get_max_threads(), 1);

    OMPutil::get_inst().set_qulacs_num_threads(1LL << 10, 10);
    ASSERT_EQ(omp_get_max_threads(), max_threads);

    OMPutil::get_inst().set_qulacs_num_threads(1LL << 9, 10);
    ASSERT_EQ(omp_get_max_threads(), 1);

    OMPutil::get_inst().set_qulacs_num_threads(1LL << 34, 34);
    ASSERT_EQ(omp_get_max_threads(), max_threads);

    OMPutil::get_inst().set_qulacs_num_threads(1LL << 33, 34);
    ASSERT_EQ(omp_get_max_threads(), 1);

    OMPutil::get_inst().reset_qulacs_num_threads();
    ASSERT_EQ(omp_get_max_threads(), max_threads);
}
#endif
