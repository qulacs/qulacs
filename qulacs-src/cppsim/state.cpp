


#include "state.hpp"

#ifndef _MSC_VER
extern "C" {
#include <csim/stat_ops.h>
}
#else
#include <csim/stat_ops.h>
#endif
#include <iostream>

namespace state {
    CPPCTYPE inner_product(const QuantumState* state1, const QuantumState* state2) {
		if (state1->qubit_count != state2->qubit_count) {
			std::cerr << "Error: inner_product(const QuantumState*, const QuantumState*): invalid qubit count" << std::endl;
			return 0.;
		}
		
		return state_inner_product(state1->data_c(), state2->data_c(), state1->dim);
    }
}
