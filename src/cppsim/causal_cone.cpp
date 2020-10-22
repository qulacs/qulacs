#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include "causal_cone.hpp"

std::vector<QuantumCircuit> CausalConeGenerate(const QuantumCircuit &init_circuit, 
    const std::vector<UINT> &observal_index_list)
{
    const int gate_count = init_circuit.gate_list.size();
    const int qubit_count = init_circuit.qubit_count;
    UnionFind uf(gate_count + qubit_count);
    std::vector<bool> use_qubit(qubit_count);
    for(auto observable_index : observal_index_list) {
        use_qubit[observable_index] = true;
    }

    auto gate_connect=[&](int gate_index, std::vector<UINT> &qubit_index_list)->void{
        for(auto &qubit_index:qubit_index_list){
            if(use_qubit[qubit_index]){
                uf.connect(gate_index, gate_count + qubit_index);
            }
        }
    };
    for(int i = gate_count - 1; i >= 0; i--){
        auto target_indexes = init_circuit.gate_list[i]->get_target_index_list();
        auto control_indexes = init_circuit.gate_list[i]->get_control_index_list();
        gate_connect(i,target_indexes);
        gate_connect(i,control_indexes);
        for(auto target_index:target_indexes){
            for(auto control_index:control_indexes){
                if(use_qubit[target_index] or use_qubit[control_index]){
                    uf.connect(target_index,control_index);
                    use_qubit[control_index] = use_qubit[target_index] = true;
                }
            }
        }
    }

    UINT circuit_count = 0;
    std::vector<UINT> enc(qubit_count + gate_count, -1);
    for(int i = 0; i < qubit_count; i++){
        if(use_qubit[i] and enc[uf.root(i + gate_count)] == -1){
            enc[uf.root(i + gate_count)] = circuit_count;
            circuit_count += 1;
        }
    }
    std::vector<QuantumCircuit> circuits(circuit_count(qubit_count));
    //具体的に構築する
        
}
int main(){


    QuantumState state(3);
    state.set_Haar_random_state();

    QuantumCircuit circuit(3);
    circuit.add_X_gate(0);
    circuit.add_gate(gate::CNOT(0,1));
    circuit.add_RX_gate(1,0.5);
    circuit.update_quantum_state(&state);

    Observable observable(3);
    observable.add_operator(2.0, "X 2 Y 1 Z 0");
    observable.add_operator(-3.0, "Z 2");
    std::vector<UINT> v;
    auto value = observable.get_expectation_value(&state);
    auto c = CausalConeGenerate(circuit, v);
    std::cout << value << std::endl;
    return 0;
}
