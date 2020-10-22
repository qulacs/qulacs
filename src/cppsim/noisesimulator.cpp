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
    std::vector<std::vector<UINT>> trial_gates(sample_count,std::vector<UINT>(circuit -> gate_list.size(),0));
    for(UINT i = 0;i < sample_count;++i){
        UINT gate_size = circuit ->gate_list.size();
        for(UINT q = 0;q < gate_size;++q){
            int way_choose = 4;
            if(noise_info[q].second != UINT_MAX){
                //2-qubit-gate
                way_choose *= 4;
            }
            way_choose -= 1;
            double delta = prob/(double)way_choose;
            double val = random.uniform();
            if(val<=prob){
                trial_gates[i][q] = (int)floor(val/delta)+1;
            }else{
                trial_gates[i][q] = 0;
            }
        }
    }
    std::sort(rbegin(trial_gates),rend(trial_gates));
    QuantumState Common_state(initial_state -> qubit_count);
    QuantumState Calculate_state(initial_state -> qubit_count);
    
    /*
    QuantumState IdealState(initial_state -> qubit_count);
    IdealState.load(initial_state);
    for(int i = 0;i < circuit -> gate_list.size();++i){
        circuit -> gate_list[i] -> update_quantum_state(&IdealState);
    }
    std::complex<long double> Fid = 0;
    */

    Common_state.load(initial_state);
    std::vector<UINT> result(sample_count);
    UINT done_itr = 0; // for gates i such that i < done_itr, gate i is already applied to Common_state.
    for(UINT i = 0;i < trial_gates.size();++i){
        //if noise is not applied to gate done_itr forever, we can apply gate done_itr to Common_state.

        while(done_itr < trial_gates[i].size() && trial_gates[i][done_itr] == 0){
            circuit -> gate_list[done_itr] -> update_quantum_state(&Common_state);
            done_itr++;
        }

        if(done_itr == trial_gates[i].size()){
            //if all remaining trials are same (no noise), merge them and speed up.
            UINT remaining = trial_gates.size() - i;
            std::vector<ITYPE> usage = Common_state.sampling(remaining);
            for(UINT q = 0;q < usage.size();++q){
                result[i+q] = usage[q];
            }
            //std::complex<long double> Now = state::inner_product(&Common_state,&IdealState);
            //Fid += Now*Now * (long double)(usage.size());
            break;
        }
        Calculate_state.load(&Common_state);
        result[i] = evaluate(trial_gates[i],&Calculate_state,done_itr);
        //std::complex<long double> Now = state::inner_product(&Calculate_state,&IdealState);
        //Fid += Now*Now;
    }
    //std::cout << Fid << std::endl;
    std::mt19937 Randomizer(random.int64());
    std::shuffle(begin(result),end(result),Randomizer);
    return result;
} 


UINT NoiseSimulator::evaluate(std::vector<UINT> chosen_gate,QuantumState *sampling_state,int StartPos){
    UINT gate_size = circuit -> gate_list.size();
    for(UINT q = StartPos;q < gate_size;++q){
        circuit -> gate_list[q] -> update_quantum_state(sampling_state);
        if(chosen_gate[q] != 0){
            //apply noise.
            if(chosen_gate[q] % 4 != 0){
                int choice = chosen_gate[q]%4;
                if(choice % 4 == 1){
                    //apply X gate
                    auto Xgate= gate::X(noise_info[q].first);
                    Xgate ->update_quantum_state(sampling_state);
                    delete Xgate;
                }else if(choice % 4 == 2){
                    //apply Y gate
                    auto Ygate= gate::Y(noise_info[q].first);
                    Ygate ->update_quantum_state(sampling_state);
                    delete Ygate;
                }else if(choice % 4 == 3){
                    //apply Z gate
                    auto Zgate= gate::Z(noise_info[q].first);
                    Zgate ->update_quantum_state(sampling_state);
                    delete Zgate;
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
                    delete Xgate;
                }else if(choice % 4 == 2){
                    //apply Y gate
                    auto Ygate= gate::Y(noise_info[q].second);
                    Ygate ->update_quantum_state(sampling_state);
                    delete Ygate;
                }else if(choice % 4 == 3){
                    //apply Z gate
                    auto Zgate= gate::Z(noise_info[q].second);
                    Zgate ->update_quantum_state(sampling_state);
                    delete Zgate;
                }
            }
        }
    }
    return sampling_state -> sampling(1)[0];
} 