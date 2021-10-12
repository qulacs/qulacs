#pragma once

#include <boost/range/combine.hpp>

#include "fermion_operator.hpp"
#include "observable.hpp"
#include "pauli_operator.hpp"
#include "state.hpp"
#include "type.hpp"

namespace transforms {
Observable jordan_wigner(FermionOperator const& fop);
}