#include <mpi.h>

#include <cppsim/circuit.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>
#include <cppsim/type.hpp>
#include <iostream>
#include <utility>
#include <vector>
class UnionFind {
private:
    std::vector<int> Parent;

public:
    UnionFind(int N) { Parent = std::vector<int>(N, -1); }
    int root(int A) {
        if (Parent[A] < 0) {
            return A;
        }
        return Parent[A] = root(Parent[A]);
    }

    bool same(int A, int B) { return root(A) == root(B); }

    int size(int A) { return -Parent[root(A)]; }

    bool connect(int A, int B) {
        A = root(A);
        B = root(B);
        if (A == B) {
            return false;
        }
        if (size(A) < size(B)) {
            std::swap(A, B);
        }

        Parent[A] += Parent[B];
        Parent[B] = A;

        return true;
    }
};

class DllExport Causal {
public:
    CPPCTYPE CausalConeMPI(
        const QuantumCircuit& init_circuit, const Observable& init_observable) {
        int rank = MPI::COMM_WORLD.Get_rank();
        int size = MPI::COMM_WORLD.Get_size();
        MPI::COMM_WORLD.Barrier();
        auto terms = init_observable.get_terms();
        CPPCTYPE ret;
        for (UINT i = rank; i < terms.size(); i += size) {
            auto term = terms[i];
            std::vector<UINT> observable_index_list = term->get_index_list();
            const UINT gate_count = init_circuit.gate_list.size();
            const UINT qubit_count = init_circuit.qubit_count;
            const UINT observable_count = observable_index_list.size();
            UnionFind uf(qubit_count + observable_count);
            std::vector<bool> use_qubit(qubit_count);
            std::vector<bool> use_gate(gate_count);
            for (UINT i = 0; i < observable_count; i++) {
                UINT observable_index = observable_index_list[i];
                use_qubit[observable_index] = true;
            }
            for (int i = gate_count - 1; i >= 0; i--) {
                auto target_index_list =
                    init_circuit.gate_list[i]->get_target_index_list();
                auto control_index_list =
                    init_circuit.gate_list[i]->get_control_index_list();
                for (auto target_index : target_index_list) {
                    if (use_qubit[target_index]) {
                        use_gate[i] = true;
                        break;
                    }
                }
                if (!use_gate[i]) {
                    for (auto control_index : control_index_list) {
                        if (use_qubit[control_index]) {
                            use_gate[i] = true;
                            break;
                        }
                    }
                }
                if (use_gate[i]) {
                    for (auto target_index : target_index_list) {
                        use_qubit[target_index] = true;
                    }

                    for (auto control_index : control_index_list) {
                        use_qubit[control_index] = true;
                    }

                    for (UINT i = 0; i + 1 < target_index_list.size(); i++) {
                        uf.connect(
                            target_index_list[i], target_index_list[i + 1]);
                    }
                    for (UINT i = 0; i + 1 < control_index_list.size(); i++) {
                        uf.connect(
                            control_index_list[i], control_index_list[i + 1]);
                    }
                    if (!target_index_list.empty() &&
                        !control_index_list.empty()) {
                        uf.connect(target_index_list[0], control_index_list[0]);
                    }
                }
            }
            //分解処理

            auto term_index_list = term->get_index_list();
            auto pauli_id_list = term->get_pauli_id_list();
            UINT circuit_count = 0;
            std::vector<UINT> roots;
            for (UINT i = 0; i < qubit_count; i++) {
                if (use_qubit[i] && i == uf.root(i)) {
                    roots.emplace_back(uf.root(i));
                    circuit_count += 1;
                }
            }
            std::vector<QuantumCircuit*> circuits(circuit_count, nullptr);
            CPPCTYPE expectation(1.0, 0);
            for (UINT i = 0; i < circuit_count; i++) {
                UINT root = roots[i];
                circuits[i] = new QuantumCircuit(uf.size(root));
                auto& circuit = circuits[i];
                std::vector<int> qubit_encode(qubit_count, -1);

                int idx = 0;
                for (UINT i = 0; i < qubit_count; i++) {
                    if (root == uf.root(i)) {
                        qubit_encode[i] = idx++;
                    }
                }

                for (UINT i = 0; i < gate_count; i++) {
                    if (!use_gate[i]) continue;

                    auto gate = init_circuit.gate_list[i]->copy();
                    auto target_index_list = gate->get_target_index_list();
                    auto control_index_list = gate->get_control_index_list();
                    if (uf.root(target_index_list[0]) != root) continue;
                    for (auto& idx : target_index_list) idx = qubit_encode[idx];
                    for (auto& idx : control_index_list)
                        idx = qubit_encode[idx];

                    gate->set_target_index_list(target_index_list);
                    gate->set_control_index_list(control_index_list);
                    circuit->add_gate(gate);
                }
                PauliOperator paulioperator(1.0);
                for (UINT i = 0; i < term_index_list.size(); i++) {
                    paulioperator.add_single_Pauli(
                        qubit_encode[term_index_list[i]], pauli_id_list[i]);
                }
                QuantumState state(idx);
                state.set_zero_state();
                circuit->update_quantum_state(&state);
                expectation *= paulioperator.get_expectation_value(&state);
            }

            ret += expectation * term->get_coef();
        }
        return ret;
    }
};
