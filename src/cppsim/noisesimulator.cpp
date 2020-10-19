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

NoiseSimulator::NoiseSimulator(const QuantumCircuit *init_circuit,const QuantumState *init_state){
    if(init_state == NULL){
        // initialize with zero state if not provided.
        initial_state = new QuantumState(init_circuit -> qubit_count);
        initial_state -> set_zero_state();
    }else{
        // initialize with init_state if provided.
        initial_state = init_state -> copy();
    }
    circuit = init_circuit -> copy();
    UINT n = init_circuit -> gate_list.size();
    for(UINT i = 0;i < n;++i){
        std::vector<UINT> qubit_indexs = init_circuit -> gate_list[i] -> get_target_index_list();
        for(auto x:init_circuit -> gate_list[i] -> get_control_index_list()){
            qubit_indexs.push_back(x);
        }
        if(qubit_indexs.size() == 1) qubit_indexs.push_back(UINT_MAX);
        if(qubit_indexs.size() >= 3){
            std::cerr << "Error: In NoiseSimulator gates must not over 2 qubits" << std::endl;
            return;
        }
        noise_info.push_back(std::pair<UINT,UINT>(qubit_indexs[0],qubit_indexs[1]));
    }
}


NoiseSimulator::~NoiseSimulator(){
    delete initial_state;
    delete circuit;
}

std::vector<UINT> NoiseSimulator::execute(const UINT sample_count,const double prob){
    Random random;
    std::vector<std::vector<UINT>> trial_gates;
    for(UINT i = 0;i < sample_count;++i){
        std::vector<UINT> chosen_gate;
        UINT gate_size = circuit ->gate_list.size();
        for(UINT q = 0;q < gate_size;++q){
            double now_percent = random.uniform();
            now_percent -= (1.0 - prob);
            int way_choose = 4;
            if(noise_info[q].second != UINT_MAX){
                //2-qubit-gate
                way_choose *= 4;
            }
            way_choose -= 1;
            int next_choose = std::max(0,(int)ceil(now_percent / (prob / (double)way_choose)) + 1);
            //noise_gate next_choose is chosen.
            chosen_gate.push_back(next_choose);
        }
        trial_gates.push_back(chosen_gate);
        chosen_gate.clear();
    }
    std::sort(begin(trial_gates),end(trial_gates));
    std::reverse(begin(trial_gates),end(trial_gates));
    QuantumState Common_state(initial_state -> qubit_count);
    QuantumState Calculate_state(initial_state -> qubit_count);
    Common_state.load(initial_state);
    std::vector<UINT> result(sample_count);
    int done_itr = 0; // for gates i such that i < done_itr, gate i is already applied to Common_state.

    for(int i = 0;i < trial_gates.size();++i){
        
        //if noise is not applied to gate done_itr forever, we can apply gate done_itr to Common_state.

        while(done_itr < trial_gates[i].size() && trial_gates[i][done_itr] == 0){
            circuit -> gate_list[done_itr] -> update_quantum_state(&Common_state);
            done_itr++;
        }
        Calculate_state.load(&Common_state);
        result[i] = evaluate(trial_gates[i],&Calculate_state,done_itr);
    }
    std::mt19937 Randomizer(random.int64());
    std::shuffle(begin(result),end(result),Randomizer);
    return result;
} 


UINT NoiseSimulator::evaluate(std::vector<UINT> chosen_gate,QuantumState *sampling_state,int StartPos){
    UINT gate_size = circuit -> gate_list.size();
    for(int q = StartPos;q < gate_size;++q){
        circuit -> gate_list[q] -> update_quantum_state(sampling_state);
        if(chosen_gate[q] != 0){
            //apply noise.
            if(chosen_gate[q] % 4 != 0){
                int choice = chosen_gate[q]%4;
                if(choice % 4 == 1){
                    //apply X gate
                    auto Xgate= gate::X(noise_info[q].first);
                    Xgate ->update_quantum_state(sampling_state);
                }else if(choice % 4 == 2){
                    //apply Y gate
                    auto Ygate= gate::Y(noise_info[q].first);
                    Ygate ->update_quantum_state(sampling_state);
                }else if(choice % 4 == 3){
                    //apply Z gate
                    auto Zgate= gate::Z(noise_info[q].first);
                    Zgate ->update_quantum_state(sampling_state);
                }
            }
            if(noise_info[q].second == UINT_MAX){
                //1-qubit gate don't need to apply noise anymore
                continue;
            }
            chosen_gate[q] /= 4;
            if(chosen_gate[q] % 4 != 0){
                int choice = chosen_gate[q]%4;
                if(choice % 4 == 1){
                    //apply X gate
                    auto Xgate= gate::X(noise_info[q].second);
                    Xgate ->update_quantum_state(sampling_state);
                }else if(choice % 4 == 2){
                    //apply Y gate
                    auto Ygate= gate::Y(noise_info[q].second);
                    Ygate ->update_quantum_state(sampling_state);
                }else if(choice % 4 == 3){
                    //apply Z gate
                    auto Zgate= gate::Z(noise_info[q].second);
                    Zgate ->update_quantum_state(sampling_state);
                }
            }
        }
    }
    return sampling_state -> sampling(1)[0];
} 