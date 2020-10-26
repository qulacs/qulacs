#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include "causal_cone.hpp"

using namespace std;
std::vector<QuantumCircuit *> CausalConeGenerate(const QuantumCircuit &init_circuit, 
    const std::vector<UINT> &observal_index_list)
{
    const UINT gate_count = init_circuit.gate_list.size();
    const UINT qubit_count = init_circuit.qubit_count;
    UnionFind uf(qubit_count);
    std::vector<bool> use_qubit(qubit_count);
    std::vector<bool> use_gate(gate_count);
    for(auto observable_index : observal_index_list) {
        use_qubit[observable_index] = true;
    }
    for(int i = gate_count - 1; i >= 0; i--){
        auto target_index_list = init_circuit.gate_list[i]->get_target_index_list();
        auto control_index_list = init_circuit.gate_list[i]->get_control_index_list();
        
        for(auto target_index : target_index_list){
            if(use_qubit[target_index]) {
                use_gate[i] = true;
                break;
            }
        }
        if(!use_gate[i]){
            for(auto control_index : control_index_list){
                if(use_qubit[control_index]) {
                    use_gate[i] = true;
                    break;
                }
            }
        }
        for(UINT i = 0; i + 1 < target_index_list.size(); i++){
            uf.connect(target_index_list[i], target_index_list[i + 1]);
        }
        for(UINT i = 0; i + 1 < control_index_list.size(); i++){
            uf.connect(control_index_list[i], control_index_list[i + 1]);
        }
        if(!target_index_list.empty() and !control_index_list.empty()){
            uf.connect(target_index_list[0], control_index_list[0]);
        }
    }
    UINT circuit_count = 0;
    std::vector<UINT> enc(qubit_count, -1);
    for(UINT i = 0; i < qubit_count; i++){
        if(use_qubit[i] and enc[uf.root(i)] == -1){
            enc[uf.root(i)] = circuit_count;
            circuit_count += 1;
        }
    }
    std::vector<QuantumCircuit*> circuits(circuit_count, nullptr);
    for(UINT i = 0; i < circuit_count; i++) circuits[i] = new QuantumCircuit(qubit_count);

    for(UINT i = 0; i < gate_count; i++){
        if(!use_gate[i]) continue;
        auto gate = init_circuit.gate_list[i];
        UINT circuit_index = enc[uf.root(gate->get_target_index_list()[0])];
        circuits[circuit_index]->add_gate_copy(gate);
    }
    return circuits;
}