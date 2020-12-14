#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <mpi.h>

#include "cppsim/circuit.hpp"
#include "cppsim/gate_factory.hpp"
#include "cppsim/gate_merge.hpp"
#include "cppsim/state.hpp"
#include "noisesimulatorMPI.hpp"
#include "utils.hpp"
/**
 * \~japanese-en 回路にノイズを加えてサンプリングするクラス
 */

NoiseSimulatorMPI::NoiseSimulatorMPI(const QuantumCircuit *init_circuit,
                                     const QuantumState *init_state,
                                     const std::vector<UINT> *Noise_itr) {
  if (init_state == NULL) {
    // initialize with zero state if not provided.
    initial_state = new QuantumState(init_circuit->qubit_count);
    initial_state->set_zero_state();
  } else {
    // initialize with init_state if provided.
    initial_state = init_state->copy();
  }
  circuit = init_circuit->copy();
  UINT n = init_circuit->gate_list.size();
  for (UINT i = 0; i < n; ++i) {
    std::vector<UINT> qubit_indexs =
        init_circuit->gate_list[i]->get_target_index_list();
    for (auto x : init_circuit->gate_list[i]->get_control_index_list()) {
      qubit_indexs.push_back(x);
    }
    if (qubit_indexs.size() == 1)
      qubit_indexs.push_back(UINT_MAX);
    if (qubit_indexs.size() >= 3) {
      std::cerr << "Error: In NoiseSimulator gates must not over 2 qubits"
                << std::endl;
      return;
    }
    noise_info.push_back(
        std::pair<UINT, UINT>(qubit_indexs[0], qubit_indexs[1]));
  }
  if (Noise_itr != NULL) {
    for (UINT q = 0; q < Noise_itr->size(); ++q) {
      // update them so that Noise will not be applied.
      noise_info[Noise_itr->at(q)].first = UINT_MAX;
      noise_info[Noise_itr->at(q)].second = UINT_MAX;
    }
  }
}

NoiseSimulatorMPI::~NoiseSimulatorMPI() {
  delete initial_state;
  delete circuit;
}

std::vector<UINT> NoiseSimulatorMPI::execute(const UINT sample_count,
                                             const double prob) {
  Random random;
  int myrank, numprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  std::vector<std::pair<std::vector<UINT>, UINT>>
      sampling_required_thisnode; // pair<trial_gate, number of samplings>
  std::vector<std::vector<UINT>> sendings(numprocs);

  int OneSampling_DataSize = (int)(circuit->gate_list.size() + 1);

  std::vector<UINT> sampling_required_rec;

  if (myrank == 0) {
    std::vector<std::vector<UINT>> trial_gates(
        sample_count, std::vector<UINT>(circuit->gate_list.size(), 0));
    for (UINT i = 0; i < sample_count; ++i) {
      UINT gate_size = circuit->gate_list.size();
      for (UINT q = 0; q < gate_size; ++q) {
        if (noise_info[q].first == UINT_MAX)
          continue;
        int way_choose = 4;
        if (noise_info[q].second != UINT_MAX) {
          // 2-qubit-gate
          way_choose *= 4;
        }
        way_choose -= 1;
        double delta = prob / (double)way_choose;
        double val = random.uniform();
        if (val <= prob) {
          trial_gates[i][q] = (int)floor(val / delta) + 1;
        } else {
          trial_gates[i][q] = 0;
        }
      }
    }

    std::sort(begin(trial_gates), end(trial_gates));

    int cnter_samplings = 0;
    std::vector<std::pair<std::vector<UINT>, int>> sampling_required_all;
    int cnter = 0;
    for (UINT i = 0; i < sample_count; ++i) {
      cnter_samplings++;
      if (i + 1 == sample_count or trial_gates[i] != trial_gates[i + 1]) {
        std::vector<UINT> now;
        int ok = 1;
        for (int q = 0; q < trial_gates[i].size(); ++q) {
          if (trial_gates[i][q] != 0 and ok == 1) {
            cnter += trial_gates[i].size() - q;
            ok = 0;
          }
          now.push_back(trial_gates[i][q]);
        }
        sampling_required_all.push_back(std::make_pair(now, cnter_samplings));
        cnter_samplings = 0;
      }
    }
    cnter /= numprocs;
    int now_stock = 0;
    std::vector<std::vector<std::pair<std::vector<UINT>, int>>> targets(
        numprocs);
    int itr = 0;
    for (int i = 0; i < sampling_required_all.size(); ++i) {
      int nows = 0;
      for (int q = 0; q < sampling_required_all[i].first.size(); ++q) {
        if (sampling_required_all[i].first[q] != 0) {
          nows = trial_gates[i].size() - q;
          break;
        }
      }
      now_stock += nows;
      targets[itr].push_back(sampling_required_all[i]);
      if (now_stock > cnter) {
        now_stock = 0;
        itr++;
        itr = std::min(itr, numprocs - 1);
      }
    }
    // transform vector<pair<vector<UINT>,UINT>> into vector<UINT>
    for (int i = 0; i < numprocs; ++i) {
      for (int q = 0; q < targets[i].size(); ++q) {
        std::vector<UINT> noise = targets[i][q].first;
        for (int t = 0; t < noise.size(); ++t) {
          sendings[i].push_back(noise[t]);
        }
        sendings[i].push_back(targets[i][q].second);
      }
      if (i == 0)
        sampling_required_rec = sendings[i];
      else
        Utility::send_vector(0, i, 0, sendings[i]);
    }
  } else {
    Utility::receive_vector(myrank, 0, 0, sampling_required_rec);
  }

  // now sampling required to perform in each node is in sampling_required_rec.
  // lets transform  vector<UINT> into vector<pair<vector<UINT>,UINT>> again.

  sampling_required_thisnode.resize(sampling_required_rec.size() /
                                    OneSampling_DataSize);
  for (int i = 0; i < sampling_required_rec.size(); ++i) {
    if (i % OneSampling_DataSize == OneSampling_DataSize - 1) {
      sampling_required_thisnode[i / OneSampling_DataSize].second =
          sampling_required_rec[i];
    } else {
      sampling_required_thisnode[i / OneSampling_DataSize].first.push_back(
          sampling_required_rec[i]);
    }
  }

  sort(rbegin(sampling_required_thisnode), rend(sampling_required_thisnode));
  QuantumState Common_state(initial_state->qubit_count);
  QuantumState Calculate_state(initial_state->qubit_count);

  Common_state.load(initial_state);
  std::vector<UINT> result(sample_count);
  auto result_itr = result.begin();
  UINT done_itr = 0; // for gates i such that i < done_itr, gate i is already
                     // applied to Common_state.
  for (UINT i = 0; i < sampling_required_thisnode.size(); ++i) {
    // if noise is not applied to gate done_itr forever, we can apply gate
    // done_itr to Common_state.
    std::vector<UINT> trial = sampling_required_thisnode[i].first;
    while (done_itr < trial.size() && trial[done_itr] == 0) {
      circuit->gate_list[done_itr]->update_quantum_state(&Common_state);
      done_itr++;
    }
    // recalculate is required.
    Calculate_state.load(&Common_state);
    evaluate_gates(trial, &Calculate_state, done_itr);
    std::vector<ITYPE> samples =
        Calculate_state.sampling(sampling_required_thisnode[i].second);
    for (UINT q = 0; q < samples.size(); ++q) {
      *result_itr = samples[q];
      result_itr++;
    }
  }
  while (result_itr - result.begin() != result.size())
    result.pop_back();
  std::mt19937 Randomizer(random.int64());
  std::shuffle(begin(result), end(result), Randomizer);

  // send sampling result to node 0
  for (int i = 1; i < numprocs; ++i) {
    Utility::send_vector(i, 0, 0, result);
  }
  std::vector<UINT> merged_result;
  if (myrank == 0) {
    // gather data
    for (int i = 0; i < numprocs; ++i) {
      std::vector<UINT> now_ans;
      if (i == 0)
        now_ans = result;
      else
        Utility::receive_vector(0, i, 0, now_ans);
      for (int q = 0; q < now_ans.size(); ++q) {
        merged_result.push_back(now_ans[q]);
      }
    }
  }

  return merged_result;
}

void NoiseSimulatorMPI::evaluate_gates(const std::vector<UINT> chosen_gate,
                                       QuantumState *sampling_state,
                                       const int StartPos) {
  UINT gate_size = circuit->gate_list.size();
  for (UINT q = StartPos; q < gate_size; ++q) {
    circuit->gate_list[q]->update_quantum_state(sampling_state);
    if (chosen_gate[q] != 0) {
      // apply noise.
      int chosen_val = chosen_gate[q];
      if (chosen_val <= 3) {
        // only applies to First Qubit.
        if (chosen_val % 4 == 1) {
          // apply X gate
          auto Xgate = gate::X(noise_info[q].first);
          Xgate->update_quantum_state(sampling_state);
          delete Xgate;
        } else if (chosen_val % 4 == 2) {
          // apply Y gate
          auto Ygate = gate::Y(noise_info[q].first);
          Ygate->update_quantum_state(sampling_state);
          delete Ygate;
        } else if (chosen_val % 4 == 3) {
          // apply Z gate
          auto Zgate = gate::Z(noise_info[q].first);
          Zgate->update_quantum_state(sampling_state);
          delete Zgate;
        }
      } else if (chosen_val % 4 == 0) {
        // only applies to Second Qubit.
        chosen_val /= 4;
        if (chosen_val % 4 == 1) {
          // apply X gate
          auto Xgate = gate::X(noise_info[q].second);
          Xgate->update_quantum_state(sampling_state);
          delete Xgate;
        } else if (chosen_val % 4 == 2) {
          // apply Y gate
          auto Ygate = gate::Y(noise_info[q].second);
          Ygate->update_quantum_state(sampling_state);
          delete Ygate;
        } else if (chosen_val % 4 == 3) {
          // apply Z gate
          auto Zgate = gate::Z(noise_info[q].second);
          Zgate->update_quantum_state(sampling_state);
          delete Zgate;
        }
      } else {
        // applies to both First and Second Qubit.
        auto gate_pauli =
            gate::Pauli({noise_info[q].first, noise_info[q].second},
                        {(UINT)chosen_val % 4, (UINT)chosen_val / 4});
        auto gate_dense = gate::to_matrix_gate(gate_pauli);
        gate_dense->update_quantum_state(sampling_state);
        delete gate_pauli;
        delete gate_dense;
      }
    }
  }
}