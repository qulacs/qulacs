#include <gtest/gtest.h>

#include <cppsim/circuit.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/noisesimulator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/general_quantum_operator.hpp>

#include "../util/util.hpp"

TEST(NoisyEvolutionTest, simple_check) {
    // just check runtime error
    UINT n = 4;
    QuantumCircuit circuit(n);
    Observable hamiltonian(n);
    hamiltonian.add_operator(1., "Z 0 Z 1");
    GeneralQuantumOperator op(n), op2(n);
    std::vector<GeneralQuantumOperator*> c_ops;
    op.add_operator(1., "Z 0");
    op2.add_operator(1., "Z 1");
    c_ops.push_back(&op);
    c_ops.push_back(&op2);
    double time = 1.;
    double dt = .01;
    auto gate = gate::NoisyEvolution(&hamiltonian, c_ops, time, dt);
    circuit.add_gate(gate);
    QuantumState state(n);
    circuit.update_quantum_state(&state);
}

TEST(NoisyEvolutionTest, unitary_evolution) {
    // check unitary evolution under ZZ hamiltonian
    UINT n = 2;
    QuantumCircuit circuit(n), circuit_ref(n);
    Observable observable(n);
    observable.add_operator(1, "X 0");
    
    // create hamiltonian and collapse operator
    Observable hamiltonian(n);
    hamiltonian.add_operator(1., "Z 0 Z 1");
    GeneralQuantumOperator op(n);
    std::vector<GeneralQuantumOperator*> c_ops;
    op.add_operator(0., "Z 0");
    c_ops.push_back(&op);
    int n_steps = 10;
    double time = 3.14/n_steps; //evolve by exp(-i*pi*ZZ/n_steps)
    double dt = .001;
    auto gate = gate::NoisyEvolution(&hamiltonian, c_ops, time, dt);
    circuit.add_gate(gate);
    
    // reference circuit
    std::vector<UINT> target_index_list{0,1};
    std::vector<UINT> pauli_id_list{3,3};
    circuit_ref.add_multi_Pauli_rotation_gate(target_index_list,pauli_id_list, -time*2);

    QuantumState state(n);
    QuantumState state_ref(n);
    gate::H(0)->update_quantum_state(&state);
    gate::H(0)->update_quantum_state(&state_ref);
    for (int k=0; k<n_steps; k++){
        circuit.update_quantum_state(&state);
        circuit_ref.update_quantum_state(&state_ref);
        auto exp = observable.get_expectation_value(&state);
        auto exp_ref = observable.get_expectation_value(&state_ref);
        ASSERT_NEAR(exp.real(), exp_ref.real(), 1e-6);
    }
}

TEST(NoisyEvolutionTest, error_scaling) {
    // check runge kutta error for unitary evolution under ZZ hamiltonian
    int n_dt = 4;
    double dt_start = 0.1; // dt will be decreased by 1/dt_scale
    double dt_scale = 1.189207115; // this choice should make the error 1/2 for each iteration
    double time = 1.;
    UINT n = 2;
    Observable observable(n);
    observable.add_operator(1, "X 0");
    
    // create hamiltonian and collapse operator
    Observable hamiltonian(n);
    hamiltonian.add_operator(1., "Z 0 Z 1");
    GeneralQuantumOperator op(n);
    std::vector<GeneralQuantumOperator*> c_ops;
    op.add_operator(0., "Z 0");
    c_ops.push_back(&op);
    // reference circuit
    QuantumCircuit circuit_ref(n);
    std::vector<UINT> target_index_list{0,1};
    std::vector<UINT> pauli_id_list{3,3};
    circuit_ref.add_multi_Pauli_rotation_gate(target_index_list,pauli_id_list, -time*2);

    double prev_error = 1;
    auto dt = dt_start;
    for (int k=0; k<n_dt; k++){
        QuantumState state(n);
        QuantumState state_ref(n);
        gate::H(0)->update_quantum_state(&state);
        gate::H(0)->update_quantum_state(&state_ref);    
        QuantumCircuit circuit(n);
        auto gate = gate::NoisyEvolution(&hamiltonian, c_ops, time, dt);
        circuit.add_gate(gate);        
        circuit.update_quantum_state(&state);
        circuit_ref.update_quantum_state(&state_ref);
        state.add_state_with_coef(-1., &state_ref);
        if (k!=0) {
            ASSERT_NEAR(std::sqrt(state.get_squared_norm())/prev_error, 1./std::pow(dt_scale, 4.), .1);
        }
        prev_error = std::sqrt(state.get_squared_norm());
        dt /= dt_scale;
    }
}

