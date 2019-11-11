#include <gtest/gtest.h>
#include "../util/util.h"
#include <gpusim/memory_ops.h>
#include <gpusim/util_func.h>

TEST(MemoryOperationTest, AllocateAndRelease) {
    const UINT n = 10;
    const ITYPE dim = 1ULL << n;
	int ngpus = get_num_device();
	for (int i = 0; i < ngpus; ++i) {
		set_device(i);
		auto ptr = allocate_quantum_state_host(dim, i);
		release_quantum_state_host(ptr, i);
	}
}

TEST(MemoryOperationTest, MemoryZeroCheck) {
    const UINT n = 10;
    const ITYPE dim = 1ULL << n;
    const double eps = 1e-14;

	int ngpus = get_num_device();
	for (int i = 0; i < ngpus; ++i) {
		set_device(i);
		auto stream_ptr = allocate_cuda_stream_host(1, i);
		auto ptr = allocate_quantum_state_host(dim, i);
		initialize_quantum_state_host(ptr, dim, stream_ptr, i);

		CTYPE* cpu_state = (CTYPE*)malloc(sizeof(CTYPE) * dim);
		get_quantum_state_host(ptr, cpu_state, dim, stream_ptr, i);
		for (ITYPE ind = 0; ind < dim; ++ind) {
			if (ind == 0) ASSERT_NEAR(cabs(cpu_state[ind] - 1.), 0., eps);
			else ASSERT_NEAR(cabs(cpu_state[ind]), 0, eps);
		}
		free(cpu_state);
		release_quantum_state_host(ptr, i);
		release_cuda_stream_host(stream_ptr, 1, i);
	}
}

TEST(MemoryOperationTest, HaarRandomState) {
    const UINT n = 10;
    const ITYPE dim = 1ULL << n;
	int ngpus = get_num_device();
	for (int i = 0; i < ngpus; ++i) {
		set_device(i);
		auto stream_ptr = allocate_cuda_stream_host(1, i);
		auto ptr = allocate_quantum_state_host(dim, i);
		initialize_Haar_random_state_host(ptr, dim, stream_ptr, i);
		release_quantum_state_host(ptr, i);
		release_cuda_stream_host(stream_ptr, 1, i);
	}
}

TEST(MemoryOperationTest, LargeMemory) {
    const UINT n = 20;
    const ITYPE dim = 1ULL << n;
	int ngpus = get_num_device();
	for (int i = 0; i < ngpus; ++i) {
		set_device(i);
		auto stream_ptr = allocate_cuda_stream_host(1, i);
		auto ptr = allocate_quantum_state_host(dim, i);
		initialize_quantum_state_host(ptr, dim, stream_ptr, i);
		release_quantum_state_host(ptr, i);
		release_cuda_stream_host(stream_ptr, 1, i);
	}
}
