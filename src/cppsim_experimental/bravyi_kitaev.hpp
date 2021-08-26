#pragma once

#include <boost/range/combine.hpp>

#include "fermion_operator.hpp"
#include "observable.hpp"
#include "pauli_operator.hpp"
#include "state.hpp"
#include "type.hpp"

namespace transforms {
Observable bravyi_kitaev(FermionOperator const& fop, UINT n_qubits);
UINT count_qubits(FermionOperator const& fop);
Observable inline_sum(std::vector<Observable> summands, Observable seed);
Observable inline_product(std::vector<Observable> factors, Observable seed);
std::set<UINT> _update_set(UINT index, UINT n_qubits);
std::set<UINT> _occupation_set(UINT index);
std::set<UINT> _parity_set(UINT index);
Observable _transform_ladder_operator(
    UINT target_index, UINT action_id, UINT n_qubits);
Observable _transform_operator_term(
    SingleFermionOperator& sfop, CPPCTYPE coef, UINT n_qubits);
}  // namespace transforms