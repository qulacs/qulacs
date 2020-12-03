#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "state.hpp"
#include "gate_factory.hpp"
#include "gate_merge.hpp"
#include "circuit.hpp"
#include "noisesimulator.hpp"
#include <algorithm>
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
            if(noise_info[q].first == UINT_MAX) continue;
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

    std::sort(trial_gates.rbegin(),trial_gates.rend());

    std::vector<std::pair<std::vector<UINT>,UINT>> sampling_required; // pair<trial_gate, number of samplings>
    int cnter_samplings = 0;
    for(UINT i = 0;i < sample_count;++i){
        cnter_samplings++;
        if(i+1 == sample_count or trial_gates[i] != trial_gates[i+1]){
            sampling_required.push_back(std::make_pair(trial_gates[i],cnter_samplings));
            cnter_samplings = 0;
        }
    }

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
    auto result_itr = result.begin();
    UINT done_itr = 0; // for gates i such that i < done_itr, gate i is already applied to Common_state.


    for(UINT i = 0;i < sampling_required.size();++i){
        //if noise is not applied to gate done_itr forever, we can apply gate done_itr to Common_state.
        std::vector<UINT> trial = sampling_required[i].first;
        while(done_itr < trial.size() && trial[done_itr] == 0){
            circuit -> gate_list[done_itr] -> update_quantum_state(&Common_state);
            done_itr++;
        }
        //recalculate is required.
        Calculate_state.load(&Common_state);
        evaluate_gates(trial,&Calculate_state,done_itr); 
        std::vector<ITYPE> samples = Calculate_state.sampling(sampling_required[i].second);
        //std::complex<long double> Now = state::inner_product(&Calculate_state,&IdealState);
        for(UINT q = 0;q < samples.size();++q){
            *result_itr = samples[q];
            result_itr++;
            //Fid += Now*Now;
        }
    }
    //std::cout << Fid << std::endl;
    std::mt19937 Randomizer(random.int64());
    std::shuffle(begin(result),end(result),Randomizer);
    
    return result;
} 


void NoiseSimulator::evaluate_gates(const std::vector<UINT> chosen_gate,QuantumState *sampling_state,const int StartPos){
    UINT gate_size = circuit -> gate_list.size();
    for(UINT q = StartPos;q < gate_size;++q){
        circuit -> gate_list[q] -> update_quantum_state(sampling_state);
        if(chosen_gate[q] != 0){
            //apply noise.           
            int chosen_val = chosen_gate[q];
            if(chosen_val <= 3){
                //only applies to First Qubit.
                if(chosen_val % 4 == 1){
                    //apply X gate
                    auto Xgate= gate::X(noise_info[q].first);
                    Xgate -> update_quantum_state(sampling_state);
                    delete Xgate;
                }else if(chosen_val % 4 == 2){
                    //apply Y gate
                    auto Ygate= gate::Y(noise_info[q].first);
                    Ygate ->update_quantum_state(sampling_state);
                    delete Ygate;
                }else if(chosen_val % 4 == 3){
                    //apply Z gate
                    auto Zgate= gate::Z(noise_info[q].first);
                    Zgate ->update_quantum_state(sampling_state);
                    delete Zgate;
                }
            }else if(chosen_val % 4 == 0){
                //only applies to Second Qubit.
                chosen_val /= 4;
                if(chosen_val % 4 == 1){
                    //apply X gate
                    auto Xgate= gate::X(noise_info[q].second);
                    Xgate -> update_quantum_state(sampling_state);
                    delete Xgate;
                }else if(chosen_val % 4 == 2){
                    //apply Y gate
                    auto Ygate= gate::Y(noise_info[q].second);
                    Ygate ->update_quantum_state(sampling_state);
                    delete Ygate;
                }else if(chosen_val % 4 == 3){
                    //apply Z gate
                    auto Zgate= gate::Z(noise_info[q].second);
                    Zgate ->update_quantum_state(sampling_state);
                    delete Zgate;
                }
            }else{
                //applies to both First and Second Qubit.
                auto gate_pauli = gate::Pauli({noise_info[q].first,noise_info[q].second},{(UINT)chosen_val % 4,(UINT)chosen_val / 4});
                auto gate_dense = gate::to_matrix_gate(gate_pauli);
                gate_dense -> update_quantum_state(sampling_state);
                delete gate_pauli;
                delete gate_dense;
            }
        }
    }
} 
