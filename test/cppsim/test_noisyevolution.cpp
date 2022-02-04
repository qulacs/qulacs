#include <gtest/gtest.h>

#include <cppsim/circuit.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_noisy_evolution.hpp>
#include <cppsim/general_quantum_operator.hpp>
#include <cppsim/noisesimulator.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>

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
    double time = 3.14 / n_steps;  // evolve by exp(-i*pi*ZZ/n_steps)
    double dt = .001;
    auto gate = gate::NoisyEvolution(&hamiltonian, c_ops, time, dt);
    circuit.add_gate(gate);

    // reference circuit
    std::vector<UINT> target_index_list{0, 1};
    std::vector<UINT> pauli_id_list{3, 3};
    circuit_ref.add_multi_Pauli_rotation_gate(
        target_index_list, pauli_id_list, -time * 2);

    QuantumState state(n);
    QuantumState state_ref(n);
    gate::H(0)->update_quantum_state(&state);
    gate::H(0)->update_quantum_state(&state_ref);
    for (int k = 0; k < n_steps; k++) {
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
    double dt_start = 0.1;          // dt will be decreased by 1/dt_scale
    double dt_scale = 1.189207115;  // this choice should make the error 1/2 for
                                    // each iteration
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
    std::vector<UINT> target_index_list{0, 1};
    std::vector<UINT> pauli_id_list{3, 3};
    circuit_ref.add_multi_Pauli_rotation_gate(
        target_index_list, pauli_id_list, -time * 2);

    double prev_error = 1;
    auto dt = dt_start;
    QuantumState state(n);
    QuantumState state_ref(n);
    for (int k = 0; k < n_dt; k++) {
        state.set_zero_state();
        state_ref.set_zero_state();
        gate::H(0)->update_quantum_state(&state);
        gate::H(0)->update_quantum_state(&state_ref);
        QuantumCircuit circuit(n);
        auto gate = gate::NoisyEvolution(&hamiltonian, c_ops, time, dt);
        circuit.add_gate(gate);
        circuit.update_quantum_state(&state);
        circuit_ref.update_quantum_state(&state_ref);
        state.add_state_with_coef(-1., &state_ref);
        if (k != 0) {
            ASSERT_NEAR(std::sqrt(state.get_squared_norm()) / prev_error,
                1. / std::pow(dt_scale, 4.), .1);
        }
        prev_error = std::sqrt(state.get_squared_norm());
        dt /= dt_scale;
    }
}

TEST(NoisyEvolutionTest, EffectiveHamiltonian) {
    // 2 qubit dephasing dynamics with ZZ interaction
    double time = 1.;
    double dt = 0.01;
    UINT n = 2;
    // create hamiltonian and collapse operator
    Observable hamiltonian(n);
    hamiltonian.add_operator(1., "Z 0 Z 1");
    GeneralQuantumOperator op(n);
    std::vector<GeneralQuantumOperator*> c_ops;
    op.add_operator(1., "Z 0");
    c_ops.push_back(&op);
    GeneralQuantumOperator op2(n);
    op2.add_operator(1., "Z 1");
    c_ops.push_back(&op2);

    // noisy evolution gate
    auto gate = dynamic_cast<ClsNoisyEvolution*>(
        gate::NoisyEvolution(&hamiltonian, c_ops, time, dt));
    ASSERT_EQ(gate->get_effective_hamiltonian()->to_string(),
        "(1,0) Z 0 Z 1 + (0,-1) I");
}

TEST(NoisyEvolutionTest, dephasing) {
    // 2 qubit dephasing dynamics with ZZ interaction
    double time = 2.;
    double dt = 0.1;
    double decay_rate = 0.2;
    double hamiltonian_energy = 0.2;
    double ref = 0.5936940289967207;              // generated by qutip
    double ref_withoutnoise = 0.696706573268861;  // generated by qutip (needed
                                                  // for variance calculation)
    UINT n = 2;
    UINT n_samples = 1000;
    Observable observable(n);
    observable.add_operator(1, "X 0");
    // create hamiltonian and collapse operator
    Observable hamiltonian(n);
    hamiltonian.add_operator(hamiltonian_energy, "Z 0 Z 1");
    GeneralQuantumOperator op(n);
    std::vector<GeneralQuantumOperator*> c_ops;
    op.add_operator(decay_rate, "Z 0");
    c_ops.push_back(&op);
    GeneralQuantumOperator op2(n);
    op2.add_operator(decay_rate, "Z 1");
    c_ops.push_back(&op2);

    QuantumState state(n), init_state(n);
    QuantumCircuit circuit(n);
    gate::H(0)->update_quantum_state(&init_state);
    gate::H(1)->update_quantum_state(&init_state);
    circuit.add_gate(gate::NoisyEvolution(&hamiltonian, c_ops, time, dt));
    double exp = 0.;
    for (int k = 0; k < n_samples; k++) {
        state.load(&init_state);
        circuit.update_quantum_state(&state);
        exp += observable.get_expectation_value(&state).real() / n_samples;
    }
    // intrinsic variance is (<P>_withoutnoise^2-<P>_withnoise^2).
    // take 5-sigma for assertion. Correct code should violate this assertion by
    // probabilty of only 3e-4 %.
    auto fivesigma = 5 * sqrt(ref_withoutnoise * ref_withoutnoise - ref * ref) /
                     sqrt(n_samples);
    ASSERT_NEAR(exp, ref, fivesigma);
}

TEST(NoisyEvolutionTest, T1T2) {
    // 2 qubit dephasing dynamics with ZZ interaction
    double time = 2.;
    double dt = 0.1;
    double decay_rate_z = 0.2;
    double decay_rate_p = 0.6;
    double decay_rate_m = 0.1;
    double hamiltonian_energy = 1.;
    double ref = -0.3135191750739427;  // generated by qutip
    UINT n = 2;
    UINT n_samples = 100;
    Observable observable(n);
    observable.add_operator(1, "X 0");
    // create hamiltonian and collapse operator
    Observable hamiltonian(n);
    hamiltonian.add_operator(hamiltonian_energy, "Z 0 Z 1");
    std::vector<GeneralQuantumOperator*> c_ops;
    for (int k = 0; k < 6; k++) c_ops.push_back(new GeneralQuantumOperator(n));
    c_ops[0]->add_operator(decay_rate_z, "Z 0");
    c_ops[1]->add_operator(decay_rate_z, "Z 1");
    c_ops[2]->add_operator(decay_rate_p / 2, "X 0");
    c_ops[2]->add_operator(decay_rate_p / 2 * 1.i, "Y 0");
    c_ops[3]->add_operator(decay_rate_p / 2, "X 1");
    c_ops[3]->add_operator(decay_rate_p / 2 * 1.i, "Y 1");
    c_ops[4]->add_operator(decay_rate_m / 2, "X 0");
    c_ops[4]->add_operator(-decay_rate_m / 2 * 1.i, "Y 0");
    c_ops[5]->add_operator(decay_rate_m / 2, "X 1");
    c_ops[5]->add_operator(-decay_rate_m / 2 * 1.i, "Y 1");

    QuantumState state(n);
    QuantumCircuit circuit(n);
    circuit.add_gate(gate::NoisyEvolution(&hamiltonian, c_ops, time, dt));
    double exp = 0.;
    for (int k = 0; k < n_samples; k++) {
        state.set_zero_state();
        gate::H(0)->update_quantum_state(&state);
        gate::H(1)->update_quantum_state(&state);
        circuit.update_quantum_state(&state);
        exp += observable.get_expectation_value(&state).real() / n_samples;
    }
    std::cout << "NoisyEvolution: " << exp << " ref: " << ref << std::endl;
    ASSERT_NEAR(exp, ref, .1);
}

TEST(NoisyEvolutionTest, check_inf_occurence) {
    // test case for checking inf
    double time = 1.;
    double dt = 0.1;
    double decay_rate = 1.;
    double hamiltonian_energy = 1.;
    UINT n = 2;
    UINT n_samples = 100;
    Observable observable(n);
    observable.add_operator(1, "X 0");
    // create hamiltonian and collapse operator
    Observable hamiltonian(n);
    hamiltonian.add_operator(hamiltonian_energy, "Z 0 Z 1");
    GeneralQuantumOperator op(n);
    std::vector<GeneralQuantumOperator*> c_ops;
    op.add_operator(decay_rate, "Z 0");
    c_ops.push_back(&op);
    GeneralQuantumOperator op2(n);
    op2.add_operator(decay_rate, "Z 1");
    c_ops.push_back(&op2);

    QuantumState state(n);
    QuantumCircuit circuit(n);
    circuit.add_gate(gate::NoisyEvolution(&hamiltonian, c_ops, time, dt));
    double exp = 0.;
    for (int k = 0; k < n_samples; k++) {
        circuit.update_quantum_state(&state);
        ASSERT_FALSE(
            std::isinf(observable.get_expectation_value(&state).real()));
    }
}
