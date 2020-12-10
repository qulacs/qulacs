#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <mpi.h>

#include "cppsim/state.hpp"
#include "cppsim/gate_factory.hpp"
#include "cppsim/gate_merge.hpp"
#include "cppsim/circuit.hpp"
#include "noisesimulatorMPI.hpp"
#include "utils.hpp"
/**
 * \~japanese-en 回路にノイズを加えてサンプリングするクラス
 */

NoiseSimulatorMPI::NoiseSimulatorMPI(const QuantumCircuit *init_circuit,const QuantumState *init_state,const std::vector<UINT> *Noise_itr){
    if(init_state == NULL){
        // initialize with zero state if not provided.
        initial_state = new QuantumState(init_circuit -> qubit_count);
        initial_state -> set_zero_state();
    }else{
        // initialize with init_state if provided.
        initial_state = init_state -> copy();
    }
    circuit = init_circuit -> copy();
}


NoiseSimulatorMPI::~NoiseSimulatorMPI(){
    delete initial_state;
    delete circuit;
}

std::vector<UINT> NoiseSimulatorMPI::execute(const UINT sample_count){
	Random random;
    int myrank,numprocs;
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    std::vector<std::pair<std::vector<UINT>,UINT>> sampling_required_thisnode; // pair<trial_gate, number of samplings>
    std::vector<std::vector<UINT>> sendings(numprocs);

    int OneSampling_DataSize = (int)( circuit ->gate_list.size() + 1);

    std::vector<UINT> sampling_required_rec;

    if(myrank == 0){
        std::vector<std::vector<UINT>> trial_gates(sample_count,std::vector<UINT>(circuit -> gate_list.size(),0));
        for(UINT i = 0;i < sample_count;++i){
            UINT gate_size = circuit ->gate_list.size();
            for(UINT q = 0;q < gate_size;++q){
                auto gate = circuit -> gate_list[q];
                if(gate -> is_noise() == false) continue;
                double val = random.uniform();
                std::vector<double> itr = gate -> get_cumulative_distribution();
                auto hoge = std::lower_bound(itr.begin(),itr.end(),val);
                assert(hoge != itr.begin());
                trial_gates[i][q] = std::distance(itr.begin(),hoge) - 1;
            }
        }

        std::sort(begin(trial_gates),end(trial_gates));

        int cnter_samplings = 0;
        std::vector<std::pair<std::vector<UINT>,int>> sampling_required_all;
        int cnter = 0;
        for(UINT i = 0;i < sample_count;++i){
            cnter_samplings++;
            if(i+1 == sample_count or trial_gates[i] != trial_gates[i+1]){
                std::vector<UINT> now;
                int ok = 1;
                for(int q = 0;q < trial_gates[i].size();++q){
                    if(trial_gates[i][q] != 0 and ok == 1){
                        cnter += trial_gates[i].size() - q;
                        ok = 0;
                    }
                    now.push_back(trial_gates[i][q]);
                }
                sampling_required_all.push_back(std::make_pair(now,cnter_samplings));
                cnter_samplings = 0;
            }
        }
        cnter /= numprocs;
        int now_stock = 0;
        std::vector<std::vector<std::pair<std::vector<UINT>,int>>> targets(numprocs);
        int itr = 0;
        for(int i = 0;i < sampling_required_all.size();++i){
            int nows = 0;
            for(int q = 0;q < sampling_required_all[i].first.size();++q){
                if(sampling_required_all[i].first[q] != 0 ){
                    nows =  trial_gates[i].size() - q;
                    break;
                }
            }
            now_stock += nows;
            targets[itr].push_back(sampling_required_all[i]);
            if(now_stock > cnter){
                now_stock = 0;
                itr++;
                itr = std::min(itr,numprocs-1);
            }
        }
        // transform vector<pair<vector<UINT>,UINT>> into vector<UINT>
        for(int i = 0;i < numprocs;++i){
            for(int q =0;q < targets[i].size();++q){
                std::vector<UINT> noise = targets[i][q].first;
                for(int t = 0;t < noise.size();++t){
                    sendings[i].push_back(noise[t]);
                }
                sendings[i].push_back(targets[i][q].second);
            }
            if(i == 0) sampling_required_rec = sendings[i];
            else Utility::send_vector(0,i,0,sendings[i]);
        }
    }else{
        Utility::receive_vector(myrank,0,0,sampling_required_rec);
    }

    //now sampling required to perform in each node is in sampling_required_rec.
    //lets transform  vector<UINT> into vector<pair<vector<UINT>,UINT>> again.
    
    sampling_required_thisnode.resize(sampling_required_rec.size()/OneSampling_DataSize);
    for(int i = 0;i < sampling_required_rec.size();++i){
        if(i % OneSampling_DataSize == OneSampling_DataSize-1){
            sampling_required_thisnode[i / OneSampling_DataSize].second = sampling_required_rec[i];
        }else{
            sampling_required_thisnode[i / OneSampling_DataSize].first.push_back(sampling_required_rec[i]);
        }
    }
    
    sort(rbegin(sampling_required_thisnode),rend(sampling_required_thisnode));
    QuantumState Common_state(initial_state -> qubit_count);
    QuantumState Calculate_state(initial_state -> qubit_count);
    
    Common_state.load(initial_state);
    std::vector<UINT> result(sample_count);
    auto result_itr = result.begin();
    UINT done_itr = 0; // for gates i such that i < done_itr, gate i is already applied to Common_state.
    for(UINT i = 0;i < sampling_required_thisnode.size();++i){
        //if noise is not applied to gate done_itr forever, we can apply gate done_itr to Common_state.
        std::vector<UINT> trial = sampling_required_thisnode[i].first;
        while(done_itr < trial.size() && trial[done_itr] == 0){
            auto gate = circuit -> gate_list[done_itr];
            if(gate -> is_noise() == false){
                gate -> update_quantum_state(&Common_state);
            }else{
                gate -> get_gate_list()[trial[done_itr]] -> update_quantum_state(&Common_state);
            }
            done_itr++;
        }
        //recalculate is required.
        Calculate_state.load(&Common_state);
        evaluate_gates(trial,&Calculate_state,done_itr); 
        std::vector<ITYPE> samples = Calculate_state.sampling(sampling_required_thisnode[i].second);
        for(UINT q = 0;q < samples.size();++q){
            *result_itr = samples[q];
            result_itr++;
        }
    }
    while(result_itr - result.begin() != result.size()) result.pop_back();
    std::mt19937 Randomizer(random.int64());
    std::shuffle(begin(result),end(result),Randomizer);

    //send sampling result to node 0
    for(int i = 1;i < numprocs;++i){
        Utility::send_vector(i,0,0,result);
    }
    std::vector<UINT> merged_result;
    if(myrank == 0){
        //gather data
        for(int i = 0;i < numprocs;++i){
            std::vector<UINT> now_ans;
            if(i == 0) now_ans = result;
            else Utility::receive_vector(0,i,0,now_ans);
            for(int q = 0;q < now_ans.size();++q){
                merged_result.push_back(now_ans[q]);
            }
        }
    }

    return merged_result;
} 


void NoiseSimulatorMPI::evaluate_gates(const std::vector<UINT> chosen_gate,QuantumState *sampling_state,const int StartPos){
    UINT gate_size = circuit -> gate_list.size();
    for(UINT q = StartPos;q < gate_size;++q){
        auto gate = circuit -> gate_list[q];
        if(gate -> is_noise() == false){
            gate -> update_quantum_state(sampling_state);
        }else{
            gate -> get_gate_list()[chosen_gate[q]] -> update_quantum_state(sampling_state);
        }
    }
} 
