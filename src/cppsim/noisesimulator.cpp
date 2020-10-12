#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "state.hpp"
#include "gate_factory.hpp"
#include "gate_merge.hpp"
#include "circuit.hpp"
#include "noisesimulator.hpp"
/**
 * \~japanese-en 回路にノイズを加えてサンプリングするクラス
 */

NoiseSimulator::NoiseSimulator(const QuantumCircuit *init_circuit,const double prob,const QuantumState *init_state){

        if(init_state == NULL){
            // initialize with zero state if not provided.
            initial_state = new QuantumState(init_circuit -> qubit_count);
            initial_state -> set_zero_state();
        }else{
            // initialize with init_state if provided.
            initial_state = init_state -> copy();
        }
        circuit = new QuantumCircuit(init_circuit -> qubit_count);
        UINT n = init_circuit -> gate_list.size();
        for(UINT i = 0;i < n;++i){
            circuit -> add_gate_copy(init_circuit -> gate_list[i]);
            std::vector<UINT> qubit_indexs = init_circuit -> gate_list[i] -> get_target_index_list();
            for(auto x:init_circuit -> gate_list[i] -> get_control_index_list()){
                qubit_indexs.push_back(x);
            }
            if(qubit_indexs.size() == 1){
                circuit -> add_gate(gate::DepolarizingNoise(qubit_indexs[0],prob));
            }else if(qubit_indexs.size() == 2){
                circuit -> add_gate(gate::TwoQubitDepolarizingNoise(qubit_indexs[0],qubit_indexs[1],prob));
            }else{
                std::cerr << "Error: In NoiseSimulator gates must not over 2 qubits" << std::endl;
                std::cerr << "Added nothing on gate " << i << " ." << std::endl;
            }
        }
}
		

NoiseSimulator::~NoiseSimulator(){
    delete initial_state;
    delete circuit;
}

std::vector<UINT> NoiseSimulator::execute(const UINT sample_count){
    std::vector<UINT> result;
    QuantumState sampling_state(initial_state -> qubit_count);
    for(UINT i = 0;i < sample_count;++i){
        sampling_state.load(initial_state);
        circuit -> update_quantum_state(&sampling_state);
        result.push_back(sampling_state.sampling(1)[0]);
    }
    return result;
}