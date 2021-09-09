#pragma once

#include <boost/range/combine.hpp>

#include "observable.hpp"
#include "pauli_operator.hpp"
#include "state.hpp"
#include "type.hpp"
#include "fermion_operator.hpp"

namespace transforms{
    Observable jordan_wigner(FermionOperator const& fop);
}