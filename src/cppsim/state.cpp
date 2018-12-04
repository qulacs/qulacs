


#include "state.hpp"

#ifndef _MSC_VER
extern "C" {
#include <csim/stat_ops.h>
}
#else
#include <csim/stat_ops.h>
#endif

namespace state {
    CPPCTYPE inner_product(const QuantumState* state1, const QuantumState* state2) {
        return state_inner_product(state1->data_c(), state2->data_c(), state1->dim);
    }
}