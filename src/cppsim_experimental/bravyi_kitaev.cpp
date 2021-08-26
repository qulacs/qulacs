#include "bravyi_kitaev.hpp"

namespace transforms{
Observable bravyi_kitaev(FermionOperator const& fop, UINT n_qubits) {
    UINT N = count_qubits(fop);
    if(n_qubits < N) assert(n_qubits < N && "Invalid number of qubits specified.");

    std::vector<Observable> transformed_terms;
    std::vector<SingleFermionOperator> fermion_terms = fop.get_fermion_list();
    std::vector<CPPCTYPE> coef_list = fop.get_coef_list();
    for (auto fop_tuple : boost::combine(fermion_terms, coef_list)) {
        SingleFermionOperator sfop;
        CPPCTYPE coef;
        boost::tie(sfop, coef) = fop_tuple;
        transformed_terms.push_back(_transform_operator_term(sfop, coef, n_qubits));
    }
    return inline_sum(transformed_terms, Observable());
}

UINT count_qubits(FermionOperator const& fop) {
    UINT qubits = 0;
    std::vector<SingleFermionOperator> fermion_terms = fop.get_fermion_list();
    for (auto sfop : fermion_terms) {
        auto target_index_list = sfop.get_target_index_list();
        for (auto target_index : target_index_list) {
            qubits = std::max(qubits, target_index + 1);
        }
    }
    return qubits;
}

Observable inline_sum(std::vector<Observable> summands, Observable seed) {
    Observable result = seed;
    for (auto summand : summands) {
        result += summand;
    }
    return result;
}

Observable inline_product(std::vector<Observable> factors, Observable seed) {
    Observable result = seed;
    for (auto factor : factors) {
        result *= factor;
    }
    return result;
}

std::set<UINT> _update_set(UINT index, UINT n_qubits) {
    std::set<UINT> indices;

    // For bit manipulation we need to count from 1 rather than 0
    index += 1;

    while (index <= n_qubits) {
        indices.insert(index - 1);
        // Add least significant one to index
        // E.g. 00010100 -> 00011000
        index += index & -index;
    }
    return indices;
}

std::set<UINT> _occupation_set(UINT index) {
    std::set<UINT> indices;
    // For bit manipulation we need to count from 1 rather than 0
    index += 1;

    indices.insert(index - 1);
    UINT parent = index & (index - 1);
    index -= 1;

    while (index != parent) {
        indices.insert(index - 1);
        // Remove least significant one from index
        // E.g. 00010100 -> 00010000
        index &= index - 1;
    }
    return indices;
}

std::set<UINT> _parity_set(UINT index) {
    std::set<UINT> indices;
    // For bit manipulation we need to count from 1 rather than 0
    index += 1;

    while (index > 0) {
        indices.insert(index - 1);
        // Remove least significant one from index
        // E.g. 00010100 -> 00010000
        index &= index - 1;
    }
    return indices;
}

Observable _transform_ladder_operator(
    UINT target_index, UINT action_id, UINT n_qubits) {
    std::set<UINT> update_set = _update_set(target_index, n_qubits);
    std::set<UINT> occupation_set = _occupation_set(target_index);
    std::set<UINT> parity_set = _parity_set(target_index - 1);

    // Initialize the transformed majorana operator (a_p^\dagger + a_p) / 2
    Observable transformed_operator;

    std::vector<UINT> target_index_list;
    target_index_list.insert(
        target_index_list.end(), update_set.begin(), update_set.end());
    target_index_list.insert(
        target_index_list.end(), parity_set.begin(), parity_set.end());

    std::vector<UINT> pauli_x_list(update_set.size(), PAULI_ID_X);
    std::vector<UINT> pauli_z_list(parity_set.size(), PAULI_ID_Z);
    std::vector<UINT> pauli_list;
    pauli_list.insert(pauli_list.end(), pauli_x_list.begin(), pauli_x_list.end());
    pauli_list.insert(pauli_list.end(), pauli_z_list.begin(), pauli_z_list.end());

    transformed_operator.add_term(
        0.5, MultiQubitPauliOperator(target_index_list, pauli_list));

    // Get the transformed (a_p^\dagger - a_p) / 2
    // Below is equivalent to X(update_set) * Z(parity_set ^ occupation_set)
    Observable transformed_majorana_difference;

    std::set<UINT> index_only = {target_index};
    std::set<UINT> update_minus_index_set;
    std::set_difference(update_set.begin(), update_set.end(),
        index_only.begin(), index_only.end(),
        std::inserter(update_minus_index_set, update_minus_index_set.end()));
    std::set<UINT> parity_xor_occupation_set;
    std::set_symmetric_difference(parity_set.begin(), parity_set.end(),
        occupation_set.begin(), occupation_set.end(),
        std::inserter(
            parity_xor_occupation_set, parity_xor_occupation_set.end()));
    std::set<UINT> p_xor_o_minus_index_set;
    std::set_difference(parity_xor_occupation_set.begin(),
        parity_xor_occupation_set.end(), index_only.begin(), index_only.end(),
        std::inserter(p_xor_o_minus_index_set, p_xor_o_minus_index_set.end()));

    target_index_list.clear();
    pauli_list.clear();
    pauli_x_list.clear();
    pauli_z_list.clear();

    pauli_x_list = std::vector<UINT>(update_minus_index_set.size(), PAULI_ID_X);
    pauli_z_list = std::vector<UINT>(p_xor_o_minus_index_set.size(), PAULI_ID_Z);

    target_index_list.push_back(target_index);
    target_index_list.insert(target_index_list.end(),
        update_minus_index_set.begin(), update_minus_index_set.end());
    target_index_list.insert(target_index_list.end(),
        p_xor_o_minus_index_set.begin(), p_xor_o_minus_index_set.end());

    pauli_list.push_back(PAULI_ID_Y);
    pauli_list.insert(pauli_list.end(), pauli_x_list.begin(), pauli_x_list.end());
    pauli_list.insert(pauli_list.end(), pauli_z_list.begin(), pauli_z_list.end());

    transformed_majorana_difference.add_term(
        -0.5i, MultiQubitPauliOperator(target_index_list, pauli_list));

    // Raising
    if (action_id == 1)
        transformed_operator += transformed_majorana_difference;
    else
        transformed_operator -= transformed_majorana_difference;
    return transformed_operator;
}

Observable _transform_operator_term(
    SingleFermionOperator& sfop, CPPCTYPE coef, UINT n_qubits) {
    auto target_index_list = sfop.get_target_index_list();
    auto action_id_list = sfop.get_action_id_list();
    std::vector<Observable> transformed_ladder_ops;
    for (auto ladder_operator :
        boost::combine(target_index_list, action_id_list)) {
        UINT target_index;
        UINT action_id;
        boost::tie(target_index, action_id) = ladder_operator;
        transformed_ladder_ops.push_back(
            _transform_ladder_operator(target_index, action_id, n_qubits));
    }
    Observable seed;
    seed.add_term(coef, "I 0");
    return inline_product(transformed_ladder_ops, seed);
}
}