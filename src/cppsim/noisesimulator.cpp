#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "noisesimulator.hpp"

#include <algorithm>
#include <numeric>

#include "circuit.hpp"
#include "gate_factory.hpp"
#include "gate_merge.hpp"
#include "state.hpp"
/**
 * \~japanese-en 回路にノイズを加えてサンプリングするクラス
 */

NoiseSimulator::NoiseSimulator(
    const QuantumCircuit* init_circuit, const QuantumState* init_state) {
    if (init_state == NULL) {
        // initialize with zero state if not provided.
        initial_state = new QuantumState(init_circuit->qubit_count);
        initial_state->set_zero_state();
    } else {
        // initialize with init_state if provided.
        initial_state = init_state->copy();
    }
    circuit = init_circuit->copy();
    for (UINT i = 0; i < circuit->gate_list.size(); ++i) {
        auto gate = circuit->gate_list[i];
        if (!gate->is_noise()) continue;
        dynamic_cast<QuantumGate_Probabilistic*>(gate)
            ->optimize_ProbablisticGate();
    }
}

NoiseSimulator::~NoiseSimulator() {
    delete initial_state;
    delete circuit;
}

NoiseSimulator::Result::Result(
    const std::vector<std::pair<QuantumState*, UINT>>& result_) {
    std::transform(result_.begin(), result_.end(), std::back_inserter(result),
        [](const std::pair<QuantumState*, UINT>& p) {
            return std::make_pair(p.first->copy(), p.second);
        });
}

NoiseSimulator::Result::~Result() {
    for (const auto& p : result) {
        delete p.first;
    }
}

std::vector<ITYPE> NoiseSimulator::Result::sampling() const {
    std::vector<ITYPE> sampling_result;
    for (auto& p : result) {
        std::vector<ITYPE> sampling_result_p = p.first->sampling(p.second);
        std::copy(sampling_result_p.begin(), sampling_result_p.end(),
            std::back_inserter(sampling_result));
    }
    return sampling_result;
}

std::vector<ITYPE> NoiseSimulator::execute(const UINT execution_count) {
    Result* result = execute_and_get_result(execution_count);
    std::vector<ITYPE> ret = result->sampling();
    delete result;
    return ret;
}

NoiseSimulator::Result* NoiseSimulator::execute_and_get_result(
    const UINT sample_count) {
    std::vector<SamplingRequest> sampling_required =
        generate_sampling_request(sample_count);
    std::vector<std::pair<QuantumState*, UINT>> simulate_result =
        simulate(sampling_required);
    Result* result = new Result(simulate_result);
    for (const std::pair<QuantumState*, UINT>& p : simulate_result) {
        delete p.first;
    }
    return result;
}

std::vector<NoiseSimulator::SamplingRequest>
NoiseSimulator::generate_sampling_request(const UINT sample_count) {
    std::vector<std::vector<UINT>> selected_gate_pos(
        sample_count, std::vector<UINT>(circuit->gate_list.size(), 0));

    const UINT gate_size = (UINT)circuit->gate_list.size();
    for (UINT i = 0; i < sample_count; ++i) {
        for (UINT j = 0; j < gate_size; ++j) {
            selected_gate_pos[i][j] =
                randomly_select_which_gate_pos_to_apply(circuit->gate_list[j]);
        }
    }

    std::sort(begin(selected_gate_pos), end(selected_gate_pos));
    std::reverse(begin(selected_gate_pos), end(selected_gate_pos));

    // merge sampling requests with same applied gate.
    // we don't have to recalculate same quantum state twice for sampling.
    std::vector<SamplingRequest> required_sampling_requests;
    int current_sampling_count = 0;
    for (UINT i = 0; i < sample_count; ++i) {
        current_sampling_count++;
        if (i + 1 == sample_count ||
            selected_gate_pos[i] != selected_gate_pos[i + 1]) {
            // can not merge (i-th) sampling step and (i+1-th) sampling step
            // together bacause applied gate is different.

            required_sampling_requests.push_back(
                SamplingRequest(selected_gate_pos[i], current_sampling_count));
            current_sampling_count = 0;
        }
    }

    return required_sampling_requests;
}

std::vector<std::pair<QuantumState*, UINT>> NoiseSimulator::simulate(
    std::vector<NoiseSimulator::SamplingRequest> sampling_requests) {
    std::sort(begin(sampling_requests), end(sampling_requests),
        [](const NoiseSimulator::SamplingRequest& l,
            const NoiseSimulator::SamplingRequest& r) {
            return l.gate_pos > r.gate_pos;
        });

    std::vector<std::pair<QuantumState*, UINT>> simulation_result;

    QuantumState common_state(initial_state->qubit_count);
    QuantumState buffer(initial_state->qubit_count);

    common_state.load(initial_state);
    UINT done_itr = 0;  // for gates i such that i < done_itr, gate i is already
                        // applied to common_state.

    for (UINT i = 0; i < sampling_requests.size(); ++i) {
        // if gate[done_itr] will always choice 0-th gate to apply to state, we
        // can apply 0-th gate of gate[done_itr] to common_state.

        std::vector<UINT> current_gate_pos = sampling_requests[i].gate_pos;
        while (done_itr < current_gate_pos.size() &&
               current_gate_pos[done_itr] == 0) {
            auto gate = circuit->gate_list[done_itr];
            if (!gate->is_noise()) {
                gate->update_quantum_state(&common_state);
            } else {
                dynamic_cast<QuantumGate_Probabilistic*>(gate)
                    ->get_gate_list()[current_gate_pos[done_itr]]
                    ->update_quantum_state(&common_state);
            }
            done_itr++;
        }

        buffer.load(&common_state);
        apply_gates(current_gate_pos, &buffer, done_itr);
        simulation_result.emplace_back(
            buffer.copy(), sampling_requests[i].num_of_sampling);
    }
    return simulation_result;
}

UINT NoiseSimulator::randomly_select_which_gate_pos_to_apply(
    QuantumGateBase* gate) {
    if (!gate->is_noise()) return 0;

    std::vector<double> current_cumulative_distribution =
        dynamic_cast<QuantumGate_Probabilistic*>(gate)
            ->get_cumulative_distribution();
    double tmp = random.uniform();
    auto gate_iterator =
        std::lower_bound(begin(current_cumulative_distribution),
            end(current_cumulative_distribution), tmp);

    // -1 is applied to gate_pos since gate_iterator is based on
    // cumulative distribution.
    auto gate_pos =
        std::distance(begin(current_cumulative_distribution), gate_iterator) -
        1;
    return gate_pos;
};

void NoiseSimulator::apply_gates(const std::vector<UINT>& chosen_gate,
    QuantumState* sampling_state, const int StartPos) {
    const UINT gate_size = (UINT)circuit->gate_list.size();
    for (UINT q = StartPos; q < gate_size; ++q) {
        auto gate = circuit->gate_list[q];
        if (!gate->is_noise()) {
            gate->update_quantum_state(sampling_state);
        } else {
            dynamic_cast<QuantumGate_Probabilistic*>(gate)
                ->get_gate_list()[chosen_gate[q]]
                ->update_quantum_state(sampling_state);
        }
    }
}
