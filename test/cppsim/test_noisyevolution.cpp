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
    // copy()が成功するかテストする。
    auto gate2 = gate->copy();
    delete gate;
    circuit.add_gate(gate2);
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
    auto h0 = gate::H(0);
    h0->update_quantum_state(&state);
    h0->update_quantum_state(&state_ref);
    delete h0;
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
    double dt = dt_start;
    QuantumState state(n);
    QuantumState state_ref(n);
    for (int k = 0; k < n_dt; k++) {
        state.set_zero_state();
        state_ref.set_zero_state();
        auto h0 = gate::H(0);
        h0->update_quantum_state(&state);
        h0->update_quantum_state(&state_ref);
        delete h0;
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
    auto gate = gate::NoisyEvolution(&hamiltonian, c_ops, time, dt);
    auto effective_hamiltonian = gate->get_effective_hamiltonian();
    ASSERT_EQ(effective_hamiltonian->to_string(), "(1,0) Z 0 Z 1 + (0,-1) ");
    delete effective_hamiltonian;
    delete gate;
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
    auto h0 = gate::H(0);
    auto h1 = gate::H(1);
    h0->update_quantum_state(&init_state);
    h1->update_quantum_state(&init_state);
    delete h0;
    delete h1;
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
    double fivesigma = 5 *
                       sqrt(ref_withoutnoise * ref_withoutnoise - ref * ref) /
                       sqrt(n_samples);
    ASSERT_NEAR(exp, ref, fivesigma);
}

std::string t1t2_test() {
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

    auto noisy_evolution_gate =
        gate::NoisyEvolution(&hamiltonian, c_ops, time, dt);
    circuit.add_gate(noisy_evolution_gate);

    double exp = 0.;
    for (int k = 0; k < n_samples; k++) {
        state.set_zero_state();
        auto h0 = gate::H(0);
        auto h1 = gate::H(1);
        h0->update_quantum_state(&state);
        h1->update_quantum_state(&state);
        delete h0;
        delete h1;
        circuit.update_quantum_state(&state);
        exp += observable.get_expectation_value(&state).real() / n_samples;
    }
    std::cout << "NoisyEvolution: " << exp << " ref: " << ref << std::endl;

    for (int k = 0; k < c_ops.size(); ++k) {
        delete c_ops[k];
    }
    return _CHECK_NEAR(exp, ref, .1);
}

TEST(NoisyEvolutionTest, T1T2) {
    int test_count = 3;
    int pass_count = 0;
    for (UINT i = 0; i < test_count; i++) {
        std::string err_message = t1t2_test();
        if (err_message == "")
            pass_count++;
        else
            std::cerr << err_message;
    }
    ASSERT_GE(pass_count, test_count - 1);
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

TEST(NoisyEvolutionTest_fast, T1T2) {
    // 2 qubit dephasing dynamics with ZZ interaction
    double time = 2.;
    double decay_rate_z = 0.2;
    double decay_rate_p = 0.6;
    double decay_rate_m = 0.1;
    double hamiltonian_energy = 1.;
    double ref = -0.3135191750739427;  // generated by qutip
    UINT n = 2;
    UINT n_samples = 400;

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

    for (auto c_op : c_ops) delete c_op;
    return;

    QuantumState state(n);
    QuantumCircuit circuit(n);
    circuit.add_gate(gate::NoisyEvolution_fast(&hamiltonian, c_ops, time));
    double exp = 0.;
    for (int k = 0; k < n_samples; k++) {
        state.set_zero_state();
        auto h0 = gate::H(0);
        auto h1 = gate::H(1);
        h0->update_quantum_state(&state);
        h1->update_quantum_state(&state);
        delete h0;
        delete h1;
        circuit.update_quantum_state(&state);
        exp += observable.get_expectation_value(&state).real() / n_samples;
    }
    std::cout << "NoisyEvolution: " << exp << " ref: " << ref << std::endl;
    ASSERT_NEAR(exp, ref, .1);
}

TEST(NoisyEvolutionTest_fast, tekitouu) {
    // 2 qubit dephasing dynamics with ZZ interaction
    double time = 0.8;
    double dt = 0.1;
    double decay_rate_z = 0.2;
    double decay_rate_p = 0.6;
    double decay_rate_m = 0.1;
    double hamiltonian_energy = 1.;
    UINT n = 2;
    UINT n_samples = 1000;

    Observable observable(n);
    observable.add_operator(1, "X 0");
    // create hamiltonian and collapse operator
    Observable hamiltonian(n);
    hamiltonian.add_operator(hamiltonian_energy, "Z 0 Z 1");
    hamiltonian.add_operator(0.5, "X 0");
    hamiltonian.add_operator(0.5, "Y 1");
    std::vector<GeneralQuantumOperator*> c_ops;
    for (int k = 0; k < 10; k++) c_ops.push_back(new GeneralQuantumOperator(n));
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
    c_ops[6]->add_operator(0.3, "X 0");
    c_ops[7]->add_operator(0.2, "Y 0");
    c_ops[8]->add_operator(0.2, "X 1");
    c_ops[8]->add_operator(0.2, "Y 1");
    c_ops[9]->add_operator(0.1, "X 0 Z 1");
    c_ops[9]->add_operator(0.1 * 1.i, "Y 0 Y 1");
    QuantumState state(n);
    QuantumCircuit circuit(n);
    circuit.add_gate(gate::NoisyEvolution_fast(&hamiltonian, c_ops, time));
    QuantumCircuit circuit_old(n);
    circuit_old.add_gate(gate::NoisyEvolution(&hamiltonian, c_ops, time, 0.1));
    for (auto c_op : c_ops) delete c_op;
    double exp = 0.;
    double exp_old = 0.;
    for (int k = 0; k < n_samples; k++) {
        state.set_zero_state();
        auto h0 = gate::H(0);
        auto h1 = gate::H(1);
        h0->update_quantum_state(&state);
        h1->update_quantum_state(&state);
        delete h0;
        delete h1;
        circuit.update_quantum_state(&state);
        exp += observable.get_expectation_value(&state).real() / n_samples;
    }

    for (int k = 0; k < n_samples; k++) {
        state.set_zero_state();
        auto h0 = gate::H(0);
        auto h1 = gate::H(1);
        h0->update_quantum_state(&state);
        h1->update_quantum_state(&state);
        delete h0;
        delete h1;
        circuit_old.update_quantum_state(&state);
        exp_old += observable.get_expectation_value(&state).real() / n_samples;
    }
    // ちなみにn_samples=10000だと,0.148801 exp_old: 0.141219 でした
    std::cout << "NoisyEvolution: " << exp << " exp_old: " << exp_old
              << std::endl;
    ASSERT_NEAR(exp, exp_old, .1);
}

TEST(NoisyEvolutionTest_fast, TX) {
    // 2 qubit dephasing dynamics with ZZ interaction
    double time = 30;
    double decay_rate_z = 0.05;
    double decay_rate_p = 0.05;
    double decay_rate_m = 0.01;
    double hamiltonian_energy = 1.;
    double ref = 0.7;  // generated by qutip?
    UINT n = 2;
    UINT n_samples = 300;

    Observable observable(n);
    observable.add_operator(1, "X 0 X 1");
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
    circuit.add_gate(gate::NoisyEvolution_fast(&hamiltonian, c_ops, time));
    for (auto c_op : c_ops) delete c_op;
    double exp = 0.;
    for (int k = 0; k < n_samples; k++) {
        state.set_zero_state();
        auto h0 = gate::H(0);
        auto h1 = gate::H(1);
        h0->update_quantum_state(&state);
        h1->update_quantum_state(&state);
        delete h0;
        delete h1;
        circuit.update_quantum_state(&state);
        exp += observable.get_expectation_value(&state).real() / n_samples;
    }
    std::cout << "NoisyEvolution: " << exp << " ref: " << ref << std::endl;
    ASSERT_NEAR(exp, ref, 0.15);
}

TEST(NoisyEvolutionTest_fast, almost_empty) {
    // 2 qubit dephasing dynamics with ZZ interaction
    double time = 30;
    double decay_rate_z = 0.05;
    double decay_rate_p = 0.05;
    double decay_rate_m = 0.01;
    double hamiltonian_energy = 1.;
    double ref = 0.8;
    UINT n = 2;
    UINT n_samples = 10;

    Observable observable(n);
    observable.add_operator(1, "X 0 X 1");
    // create hamiltonian and collapse operator
    Observable hamiltonian(n);
    std::vector<GeneralQuantumOperator*> c_ops(1);
    c_ops[0] = new GeneralQuantumOperator(n);
    c_ops[0]->add_operator(decay_rate_z, "Z 0");
    QuantumState state(n);
    QuantumCircuit circuit(n);
    circuit.add_gate(gate::NoisyEvolution_fast(&hamiltonian, c_ops, time));
    for (auto c_op : c_ops) delete c_op;
    double exp = 0.;
    for (int k = 0; k < n_samples; k++) {
        state.set_zero_state();
        auto h0 = gate::H(0);
        auto h1 = gate::H(1);
        h0->update_quantum_state(&state);
        h1->update_quantum_state(&state);
        delete h0;
        delete h1;
        circuit.update_quantum_state(&state);
        exp += observable.get_expectation_value(&state).real() / n_samples;
    }
    std::cout << "NoisyEvolution: " << exp << " ref: " << ref << std::endl;
    // 答えを求めることではなく、　エラーを吐かないのを確認するのが目的
}

TEST(NoisyEvolutionTest_fast, empty) {
    // 2 qubit dephasing dynamics with ZZ interaction
    double time = 30;
    double decay_rate_z = 0.05;
    double decay_rate_p = 0.05;
    double decay_rate_m = 0.01;
    double hamiltonian_energy = 1.;
    double ref = 1.0;
    UINT n = 2;
    UINT n_samples = 15;

    Observable observable(n);
    observable.add_operator(1, "X 0 X 1");
    // create hamiltonian and collapse operator
    Observable hamiltonian(n);

    std::vector<GeneralQuantumOperator*> c_ops;
    QuantumState state(n);
    QuantumCircuit circuit(n);
    circuit.add_gate(gate::NoisyEvolution_fast(&hamiltonian, c_ops, time));
    double exp = 0.;
    for (int k = 0; k < n_samples; k++) {
        state.set_zero_state();
        auto h0 = gate::H(0);
        auto h1 = gate::H(1);
        h0->update_quantum_state(&state);
        h1->update_quantum_state(&state);
        delete h0;
        delete h1;
        circuit.update_quantum_state(&state);
        exp += observable.get_expectation_value(&state).real() / n_samples;
    }
    std::cout << "NoisyEvolution: " << exp << " ref: " << ref << std::endl;
    ASSERT_NEAR(exp, ref, 0.05);
}

TEST(NoisyEvolutionTest, EmptyCops) {
    double gamma = 0.5474999999999999;
    double depth = 10;
    double dt = 1.0;
    double time = dt * depth;

    UINT n = 4;
    Observable observable(n);
    observable.add_operator(1, "X 0");
    Observable hamiltonian(n);
    for (UINT i = 0; i < n; i++) {
        std::string s = "X ";
        s += std::to_string(i);
        hamiltonian.add_operator(gamma, s);
    }

    // c_opsが空の場合にセグフォとなっていたので、テストを追加
    std::vector<GeneralQuantumOperator*> c_ops;

    QuantumState state(n);
    QuantumCircuit circuit(n);
    auto gate = gate::NoisyEvolution_fast(&hamiltonian, c_ops, time);
    circuit.add_gate(gate);
    for (int kai = 0; kai < 100; kai++) {
        for (int k = 0; k < n; k++) {
            state.set_computational_basis(k);
            circuit.update_quantum_state(&state);
            // セグフォとならずに期待値が取れればよい
            ASSERT_FALSE(
                std::isinf(observable.get_expectation_value(&state).real()));
        }
    }
}
