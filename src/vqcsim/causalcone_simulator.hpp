#include <cppsim/gate.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>
#include <cppsim/type.hpp>
#include <iostream>
#include <utility>
#include <vector>
#include <vqcsim/parametric_circuit.hpp>

class UnionFind {
private:
    std::vector<int> Parent;

public:
    explicit UnionFind(int N);
    int root(int A);

    bool same(int A, int B);

    int size(int A);

    bool connect(int A, int B);
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
        const Observable& _init_observable);

    ~CausalConeSimulator();

    void build();

    CPPCTYPE get_expectation_value();

    std::vector<std::vector<ParametricQuantumCircuit*>> get_circuit_list();

    std::vector<std::vector<PauliOperator>> get_pauli_operator_list();

    std::vector<CPPCTYPE> get_coef_list();
};
