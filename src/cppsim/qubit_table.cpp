#include "qubit_table.hpp"

#include "circuit.hpp"
#include "gate.hpp"
#include "gate_factory.hpp"

QubitTable::QubitTable(const UINT qubit_count)
    : _qubit_count(qubit_count),
      _p2l_table(qubit_count),
      _l2p_table(qubit_count),
      p2l(_p2l_table),
      l2p(_l2p_table) {
    std::iota(_p2l_table.begin(), _p2l_table.end(), 0);
    std::iota(_l2p_table.begin(), _l2p_table.end(), 0);
}

QubitTable::QubitTable(const QubitTable& qt)
    : _qubit_count(qt._qubit_count), p2l(_p2l_table), l2p(_l2p_table) {
    std::copy(qt.p2l.begin(), qt.p2l.end(), back_inserter(_p2l_table));
    std::copy(qt.l2p.begin(), qt.l2p.end(), back_inserter(_l2p_table));
}

UINT QubitTable::add_swap_gate(
    QuantumCircuit* circuit, UINT idx0, UINT idx1, UINT width) {
    return add_swap_gate(circuit, idx0, idx1, width, circuit->gate_list.size());
}

UINT QubitTable::add_swap_gate(
    QuantumCircuit* circuit, UINT idx0, UINT idx1, UINT width, UINT gate_pos) {
    //    LOG << "add_swap_gate(" << idx0 << "," << idx1 << "," << width << ")"
    //    << std::endl;
    if (idx0 == idx1) {
        return 0;
    }

    const UINT i = (idx0 < idx1) ? idx0 : idx1;
    const UINT j = (idx0 < idx1) ? idx1 : idx0;

    if (j + width > _qubit_count) {
        throw std::invalid_argument(
            "QubitTable::add_swap_gate() out of qubit range");
    }
    if (i + width > j) {
        throw std::invalid_argument(
            "QubitTable::add_swap_gate() overlap range");
    }

    if (width == 1) {
        circuit->add_gate(gate::SWAP(i, j), gate_pos);
    } else {
        circuit->add_gate(gate::FusedSWAP(i, j, width), gate_pos);
    }

    for (UINT w = 0; w < width; w++) {
        std::swap(_p2l_table[i + w], _p2l_table[j + w]);
        std::swap(_l2p_table[_p2l_table[i + w]], _l2p_table[_p2l_table[j + w]]);
    }

    return 1;
}

QuantumGateBase* QubitTable::rewrite_gate_qubit_indexes(
    QuantumGateBase* g) const {
    std::vector<UINT> target_index_list = g->get_target_index_list();
    for_each(target_index_list.begin(), target_index_list.end(),
        [&](UINT& x) { x = _l2p_table[x]; });
    std::vector<UINT> control_index_list = g->get_control_index_list();
    for_each(control_index_list.begin(), control_index_list.end(),
        [&](UINT& x) { x = _l2p_table[x]; });
    return g->create_gate_whose_qubit_indices_are_replaced(
        target_index_list, control_index_list);
}
