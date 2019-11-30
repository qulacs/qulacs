
#ifdef _USE_GPU

#include "state_gpu.hpp"
#include <assert.h>

namespace state {
	CPPCTYPE inner_product(const QuantumStateGpu* state_bra, const QuantumStateGpu* state_ket) {
		ITYPE dim = state_ket->dim;
		unsigned int device_number = state_ket->device_number;
		assert(dim == state_bra->dim);
		assert(device_number == state_bra->device_number);
		void* cuda_stream = allocate_cuda_stream_host(1, device_number);
		CPPCTYPE ret = inner_product_host(state_bra->data(), state_ket->data(), dim, cuda_stream, device_number);
		release_cuda_stream_host(cuda_stream, 1, device_number);
		return ret;
	}
}
#endif //_USE_GPU
