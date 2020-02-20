#include <stdio.h>
#include <algorithm>
#include <iterator>
#include "circuit_optimizer.hpp"
#include "circuit.hpp"
#include "gate.hpp"
#include "gate_merge.hpp"
#include "gate_matrix.hpp"
#include "gate_factory.hpp"


UINT QuantumCircuitOptimizer::get_rightmost_commute_index(UINT gate_index){
    UINT cursor = gate_index+1;
    for( ; cursor < circuit->gate_list.size() ; ++cursor){
        if( ! circuit->gate_list[gate_index]->is_commute(circuit->gate_list[cursor]) ) break;
    }
    return cursor-1;
}

UINT QuantumCircuitOptimizer::get_leftmost_commute_index(UINT gate_index){
    // be careful for underflow of unsigned value
    int cursor = (signed)(gate_index-1);
    for( ; cursor >= 0; cursor-- ){
        if( ! circuit->gate_list[gate_index]->is_commute(circuit->gate_list[cursor]) ) break;
    }
    return cursor+1;
}

UINT QuantumCircuitOptimizer::get_merged_gate_size(UINT gate_index1, UINT gate_index2){
    auto fetch_target_index = [](const std::vector<TargetQubitInfo>& info_list) { std::vector<UINT> index_list; for (auto val : info_list) index_list.push_back(val.index()); return index_list; };
    auto fetch_control_index = [](const std::vector<ControlQubitInfo>& info_list) { std::vector<UINT> index_list; for (auto val : info_list) index_list.push_back(val.index()); return index_list; };

    auto target_index_list1 = fetch_target_index(circuit->gate_list[gate_index1]->target_qubit_list);
    auto target_index_list2 = fetch_target_index(circuit->gate_list[gate_index2]->target_qubit_list);
    auto control_index_list1 = fetch_control_index(circuit->gate_list[gate_index1]->control_qubit_list);
    auto control_index_list2 = fetch_control_index(circuit->gate_list[gate_index2]->control_qubit_list);

	std::sort(target_index_list1.begin(), target_index_list1.end());
	std::sort(target_index_list2.begin(), target_index_list2.end());
	std::sort(control_index_list1.begin(), control_index_list1.end());
	std::sort(control_index_list2.begin(), control_index_list2.end());

    std::vector<UINT> target_index_merge, control_index_merge, whole_index;
    std::set_union(
        target_index_list1.begin(), target_index_list1.end(),
        target_index_list2.begin(), target_index_list2.end(),
        std::back_inserter(target_index_merge));
    std::set_union(
        control_index_list1.begin(), control_index_list1.end(),
        control_index_list2.begin(), control_index_list2.end(),
        std::back_inserter(control_index_merge));
    std::set_union(
        target_index_merge.begin(), target_index_merge.end(),
        control_index_merge.begin(), control_index_merge.end(),
        std::back_inserter(whole_index));
    return (UINT)(whole_index.size());
}

/////////////////////////////////////


bool QuantumCircuitOptimizer::is_neighboring(UINT gate_index1, UINT gate_index2){
    assert(gate_index1 != gate_index2);
    if(gate_index1 > gate_index2) std::swap(gate_index1, gate_index2);
    UINT ind1_right = this->get_rightmost_commute_index(gate_index1);
    UINT ind2_left = this->get_leftmost_commute_index(gate_index2);
    return ind2_left <= ind1_right + 1;
}

void QuantumCircuitOptimizer::optimize(QuantumCircuit* circuit_, UINT max_block_size){
    circuit = circuit_;
    bool merged_flag = true;
    while(merged_flag){
        merged_flag = false;
        for(UINT ind1 = 0 ; ind1 < circuit->gate_list.size() ; ++ind1){
            for(UINT ind2 = ind1+1 ; ind2 < circuit->gate_list.size() ; ++ind2){
                // parametric gate cannot be merged
                if( circuit->gate_list[ind1]->is_parametric() || circuit->gate_list[ind2]->is_parametric() ) continue;

                // if merged block size is larger than max_block_size, we cannot merge them
                if(this->get_merged_gate_size(ind1,ind2) > max_block_size) continue;

                // if they are separated by not-commutive gate, we cannot merge them
                // TODO: use cache for merging
                if(! this->is_neighboring(ind1,ind2)) continue;

                // generate merged gate
                auto merged_gate = gate::merge(circuit->gate_list[ind1], circuit->gate_list[ind2]);

                // remove merged two gates, and append new one
                UINT ind2_left = this->get_leftmost_commute_index(ind2);
                // insert at max(just after first_applied_gate, just before left-most gate commuting with later_applied_gate )
                // Insertion point is always later than the first, and always earlier than the second.
                UINT insert_point = std::max(ind2_left, ind1+1);

                // Not to change index with removal, process from later ones to earlier ones. 
                circuit->remove_gate(ind2);
                circuit->add_gate(merged_gate, insert_point);
                circuit->remove_gate(ind1);

                ind2--;
                merged_flag = true;
            }
        }
    }
}


QuantumGateMatrix* QuantumCircuitOptimizer::merge_all(const QuantumCircuit* circuit_) {
    QuantumGateBase* identity = gate::Identity(0);
    QuantumGateMatrix* current_gate = gate::to_matrix_gate(identity);
    QuantumGateMatrix* next_gate = NULL;
    delete identity;

    for (auto gate : circuit_->gate_list){
        next_gate = gate::merge(current_gate, gate);
        delete current_gate;
        current_gate = next_gate;
    }
    return current_gate;
}