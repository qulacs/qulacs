#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "noisesimulator.hpp"

#include <algorithm>

#include "circuit.hpp"
//#include "gate_factory.hpp"
//#include "gate_merge.hpp"
#include "state.hpp"
/**
 * \~japanese-en 回路にノイズを加えてサンプリングするクラス
 */

NoiseSimulator::NoiseSimulator(
    const QuantumCircuit* init_circuit, const QuantumState* init_state) {
    if (init_state == NULL) {
        // initialize with zero state if not provided.
        initial_state = new QuantumState(init_circuit->get_qubit_count());
        initial_state->set_zero_state();
    } else {
        // initialize with init_state if provided.
        initial_state = init_state->copy();
    }
    circuit = init_circuit->copy();
    for (int i = 0; i < circuit->get_gate_list().size(); ++i) {
        auto gate = circuit->get_gate_list()[i];
        if (gate->get_map_type() != Probabilistic) continue;
        gate->optimize_ProbabilisticGate();
    }
    /*
    UINT n = init_circuit -> get_gate_list().size();
    for(UINT i = 0;i < n;++i){
        std::vector<UINT> qubit_indexs = init_circuit -> get_gate_list()[i] ->
    get_target_index_list(); for(auto x:init_circuit -> get_gate_list()[i] ->
    get_control_index_list()){ qubit_indexs.push_back(x);
        }
        if(qubit_indexs.size() == 1) qubit_indexs.push_back(UINT_MAX);
        if(qubit_indexs.size() >= 3){
            std::cerr << "Error: In NoiseSimulator gates must not over 2 qubits"
    << std::endl; return;
        }
        noise_info.push_back(std::pair<UINT,UINT>(qubit_indexs[0],qubit_indexs[1]));
    }
    */
}

NoiseSimulator::~NoiseSimulator() {
    delete initial_state;
    delete circuit;
}

std::vector<UINT> NoiseSimulator::execute(const UINT sample_count) {
    Random random;
    std::vector<std::vector<UINT>> trial_gates(
        sample_count, std::vector<UINT>(circuit->get_gate_list().size(), 0));
    for (UINT i = 0; i < sample_count; ++i) {
        UINT gate_size = circuit->get_gate_list().size();
        for (UINT q = 0; q < gate_size; ++q) {
            auto gate = circuit->get_gate_list()[q];
            if (gate->get_map_type() != Probabilistic) continue;
            double val = random.uniform();
            std::vector<double> itr = gate->get_cumulative_distribution();
            auto hoge = std::lower_bound(itr.begin(), itr.end(), val);
            assert(hoge != itr.begin());
            trial_gates[i][q] = std::distance(itr.begin(), hoge) - 1;
        }
    }

    std::sort(trial_gates.rbegin(), trial_gates.rend());

    std::vector<std::pair<std::vector<UINT>, UINT>>
        sampling_required;  // pair<trial_gate, number of samplings>
    int cnter_samplings = 0;
    for (UINT i = 0; i < sample_count; ++i) {
        cnter_samplings++;
        if (i + 1 == sample_count || trial_gates[i] != trial_gates[i + 1]) {
            sampling_required.push_back(
                std::make_pair(trial_gates[i], cnter_samplings));
            cnter_samplings = 0;
        }
    }

    QuantumState Common_state(initial_state->qubit_count);
    QuantumState Calculate_state(initial_state->qubit_count);

    /*
    QuantumState IdealState(initial_state -> qubit_count);
    IdealState.load(initial_state);
    for(int i = 0;i < circuit -> get_gate_list().size();++i){
        circuit -> get_gate_list()[i] -> update_quantum_state(&IdealState);
    }
    std::complex<long double> Fid = 0;
    */
    Common_state.load(initial_state);
    std::vector<UINT> result(sample_count);
    auto result_itr = result.begin();
    UINT done_itr = 0;  // for gates i such that i < done_itr, gate i is already
                        // applied to Common_state.

    for (UINT i = 0; i < sampling_required.size(); ++i) {
        // if noise is not applied to gate done_itr forever, we can apply gate
        // done_itr to Common_state.
        std::vector<UINT> trial = sampling_required[i].first;
        while (done_itr < trial.size() && trial[done_itr] == 0) {
            auto gate = circuit->get_gate_list()[done_itr];
            if (gate->get_map_type() != Probabilistic) {
                gate->update_quantum_state(&Common_state);
            } else {
                gate->get_kraus_list()[trial[done_itr]]->update_quantum_state(
                    &Common_state);
            }
            done_itr++;
        }
        // recalculate is required.
        Calculate_state.load(&Common_state);
        evaluate_gates(trial, &Calculate_state, done_itr);
        std::vector<ITYPE> samples =
            Calculate_state.sampling(sampling_required[i].second);
        // std::complex<long double> Now =
        // state::inner_product(&Calculate_state,&IdealState);
        for (UINT q = 0; q < samples.size(); ++q) {
            *result_itr = samples[q];
            result_itr++;
            // Fid += Now*Now;
        }
    }
    // std::cout << Fid << std::endl;
    std::mt19937 Randomizer(random.int64());
    std::shuffle(begin(result), end(result), Randomizer);

    return result;
}

void NoiseSimulator::evaluate_gates(const std::vector<UINT> chosen_gate,
    QuantumState* sampling_state, const int StartPos) {
    UINT gate_size = circuit->get_gate_list().size();
    for (UINT q = StartPos; q < gate_size; ++q) {
        auto gate = circuit->get_gate_list()[q];
        if (gate->get_map_type() != Probabilistic) {
            gate->update_quantum_state(sampling_state);
        } else {
            gate->get_kraus_list()[chosen_gate[q]]->update_quantum_state(
                sampling_state);
        }
    }
}
