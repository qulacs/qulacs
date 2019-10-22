#include <gtest/gtest.h>
#include "../util/util.h"
#include <gpusim/memory_ops.h>
#include <gpusim/util_func.h>

TEST(MemoryOperationTest, AllocateAndRelease) {
    const UINT n = 10;
    const ITYPE dim = 1ULL << n;
    auto ptr = allocate_quantum_state_host(dim);
    release_quantum_state_host(ptr);
}

TEST(MemoryOperationTest, MemoryZeroCheck) {
    const UINT n = 10;
    const ITYPE dim = 1ULL << n;
    const double eps = 1e-14;

    auto ptr = allocate_quantum_state_host(dim);
    initialize_quantum_state_host(ptr,dim);

	CTYPE* cpu_state = (CTYPE*)malloc(sizeof(CTYPE)*dim);
	get_quantum_state_host(ptr, cpu_state, dim);
	for (ITYPE ind = 0; ind < dim; ++ind) {
        if(ind==0) ASSERT_NEAR(cabs(cpu_state[ind]-1.),0.,eps);
        else ASSERT_NEAR(cabs(cpu_state[ind]),0,eps);
    }
	free(cpu_state);
	release_quantum_state_host(ptr);
}

TEST(MemoryOperationTest, HaarRandomState) {
    const UINT n = 10;
    const ITYPE dim = 1ULL << n;
    auto ptr = allocate_quantum_state_host(dim);
    initialize_Haar_random_state_host(ptr, dim);
    release_quantum_state_host(ptr);
}

TEST(MemoryOperationTest, LargeMemory) {
    const UINT n = 20;
    const ITYPE dim = 1ULL << n;
    auto ptr = allocate_quantum_state_host(dim);
    initialize_quantum_state_host(ptr, dim);
    release_quantum_state_host(ptr);
}
