
#ifdef _USE_GPU

#include "state_gpu.hpp"

namespace state {
	CPPCTYPE DllExport inner_product(const QuantumStateGpu* state_bra, const QuantumStateGpu* state_ket) {
		return inner_product_host(state_bra->data(), state_ket->data(), state_ket->dim);
	}
}
#endif //_USE_GPU
