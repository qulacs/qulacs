#include <cppsim/gate.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>
#include <cppsim/type.hpp>
#include <utility>
#include <vector>
#include <vqcsim/parametric_circuit.hpp>

class UnionFind {
private:
    std::vector<int> Parent;

public:
    explicit UnionFind(int N) { Parent = std::vector<int>(N, -1); }
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

class DllExport CausalConeSimulator {
public:
    ParametricQuantumCircuit* init_circuit;
    Observable* init_observable;
    std::vector<std::vector<ParametricQuantumCircuit*>> circuit_list;
    std::vector<std::vector<PauliOperator>> pauli_operator_list;
    std::vector<CPPCTYPE> coef_list;
    bool build_run = false;
    CausalConeSimulator(const ParametricQuantumCircuit& _init_circuit,
        const Observable& _init_observable) {
        init_observable = _init_observable.copy();
        init_circuit = _init_circuit.copy();
    }
    ~CausalConeSimulator() {
        delete init_circuit;
        delete init_observable;
        for (auto& circuit1 : circuit_list) {
            for (auto* circuit2 : circuit1) {
                delete circuit2;
            }
        }
    }

    void build() {
        build_run = true;
        auto terms = init_observable->get_terms();
        for (auto term : terms) {
            std::vector<UINT> observable_index_list = term->get_index_list();
            const UINT gate_count = init_circuit->gate_list.size();
            const UINT qubit_count = init_circuit->qubit_count;
            const UINT observable_count = observable_index_list.size();
            UnionFind uf(qubit_count);
            std::vector<bool> use_qubit(qubit_count);
            std::vector<bool> use_gate(gate_count);
            for (UINT i = 0; i < observable_count; i++) {
                UINT observable_index = observable_index_list[i];
                use_qubit[observable_index] = true;
            }
            for (int i = gate_count - 1; i >= 0; i--) {
                auto target_index_list =
                    init_circuit->gate_list[i]->get_target_index_list();
                auto control_index_list =
                    init_circuit->gate_list[i]->get_control_index_list();
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

                    for (UINT j = 0; j + 1 < (UINT)target_index_list.size();
                         j++) {
                        uf.connect(
                            target_index_list[j], target_index_list[j + 1]);
                    }
                    for (UINT j = 0; j + 1 < (UINT)control_index_list.size();
                         j++) {
                        uf.connect(
                            control_index_list[j], control_index_list[j + 1]);
                    }
                    if (!target_index_list.empty() &&
                        !control_index_list.empty()) {
                        uf.connect(target_index_list[0], control_index_list[0]);
                    }
                }
            }
            // 分解処理

            auto term_index_list = term->get_index_list();
            auto pauli_id_list = term->get_pauli_id_list();
            UINT circuit_count = 0;
            std::vector<UINT> roots;
            for (UINT i = 0; i < qubit_count; i++) {
                if (use_qubit[i] && i == (UINT)uf.root(i)) {
                    roots.emplace_back(uf.root(i));
                    circuit_count += 1;
                }
            }
            std::vector<ParametricQuantumCircuit*> circuits(
                circuit_count, nullptr);
            std::vector<PauliOperator> pauli_operators(
                circuit_count, PauliOperator(1.0));
            // CPPCTYPE expectation(1.0, 0);
            for (UINT i = 0; i < circuit_count; i++) {
                UINT root = roots[i];
                circuits[i] = new ParametricQuantumCircuit(uf.size(root));
                auto& circuit = circuits[i];
                std::vector<int> qubit_encode(qubit_count, -1);

                int idx = 0;
                for (UINT j = 0; j < (UINT)qubit_count; j++) {
                    if (root == (UINT)uf.root(j)) {
                        qubit_encode[j] = idx++;
                    }
                }

                for (UINT j = 0; j < gate_count; j++) {
                    if (!use_gate[j]) continue;

                    auto gate = init_circuit->gate_list[j];
                    auto target_index_list = gate->get_target_index_list();
                    auto control_index_list = gate->get_control_index_list();
                    if ((UINT)uf.root(target_index_list[0]) != root) continue;
                    for (auto& target_idx : target_index_list)
                        target_idx = qubit_encode[target_idx];
                    for (auto& control_idx : control_index_list)
                        control_idx = qubit_encode[control_idx];

                    auto gate_replaced =
                        gate->create_gate_whose_qubit_indices_are_replaced(
                            target_index_list, control_index_list);
                    circuit->add_gate(gate_replaced);
                }
                auto& paulioperator = pauli_operators[i];
                for (UINT j = 0; j < (UINT)term_index_list.size(); j++) {
                    if ((UINT)uf.root(term_index_list[j]) != root) continue;
                    paulioperator.add_single_Pauli(
                        qubit_encode[term_index_list[j]], pauli_id_list[j]);
                }
            }
            circuit_list.emplace_back(circuits);
            pauli_operator_list.emplace_back(pauli_operators);
            coef_list.emplace_back(term->get_coef());
        }
    }

    CPPCTYPE get_expectation_value() {
        if (!build_run) build();
        CPPCTYPE ret;
        for (UINT i = 0; i < (UINT)circuit_list.size(); i++) {
            CPPCTYPE expectation(1.0, 0);
            auto& circuits = circuit_list[i];
            auto& pauli_operators = pauli_operator_list[i];
            auto& coef = coef_list[i];
            for (UINT j = 0; j < (UINT)circuits.size(); j++) {
                auto& circuit = circuits[j];
                auto& paulioperator = pauli_operators[j];
                QuantumState state(circuit->qubit_count);
                state.set_zero_state();
                circuit->update_quantum_state(&state);
                expectation *= paulioperator.get_expectation_value(&state);
            }
            ret += expectation * coef;
        }
        return ret;
    }
    std::vector<std::vector<ParametricQuantumCircuit*>> get_circuit_list() {
        return circuit_list;
    }
    std::vector<std::vector<PauliOperator>> get_pauli_operator_list() {
        return pauli_operator_list;
    }
    std::vector<CPPCTYPE> get_coef_list() { return coef_list; }
};
