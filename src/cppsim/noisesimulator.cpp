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
    std::vector<unsigned long long> seeds;
    Random random;
    for(UINT i = 0;i < sample_count;++i){
        seeds.push_back(random.int64());
    }
    //seedsを渡せば並列可能

    std::vector<UINT> result;
    QuantumState sampling_state(initial_state -> qubit_count);
    QuantumState BasicState(initial_state -> qubit_count);
    evaluate(std::vector<int>(circuit -> gate_list.size(),0),&BasicState);
    for(UINT i = 0;i < sample_count;++i){
        std::vector<int> chosen_gate;
        UINT gate_size = circuit ->gate_list.size();
        Random random;
        random.set_seed(seeds[i]);
        int sames = 1;
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
            chosen_gate.push_back(next_choose);
            if(next_choose != 0) sames = 0;
        }
        if(sames == 1){
            result.push_back(BasicState.sampling(1)[0]);
        }else{
            result.push_back(evaluate(chosen_gate,&sampling_state));
        }
    }
    return result;
} 


UINT NoiseSimulator::evaluate(std::vector<int> chosen_gate,QuantumState *sampling_state){
    sampling_state -> load(initial_state);
    UINT gate_size = circuit ->gate_list.size();
    for(int q = 0;q < gate_size;++q){
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