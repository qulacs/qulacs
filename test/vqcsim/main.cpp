#ifdef _USE_MPI
#include <gtest/gtest.h>
#include <mpi.h>

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);

    int ret = RUN_ALL_TESTS();

    MPI_Finalize();

    return ret;
}
#endif  // #ifdef _USE_MPI
