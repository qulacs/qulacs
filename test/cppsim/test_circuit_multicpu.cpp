#ifdef _USE_MPI
#include <gtest/gtest.h>

#include <cppsim/circuit.hpp>
#include <cppsim/circuit_optimizer.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/type.hpp>
#include <cppsim/utility.hpp>
#include <csim/constant.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include <utility>

#include "../util/util.hpp"

TEST(CircuitTest_multicpu, CircuitBasic) {
    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();
    const auto H = make_H();
    const auto S = make_S();
    const auto T = make_T();
    const auto sqrtX = make_sqrtX();
    const auto sqrtY = make_sqrtY();
    const auto P0 = make_P0();
    const auto P1 = make_P1();

    const UINT n = 4;
    UINT dim = 1ULL << n;
    MPIutil& mpiutil = MPIutil::get_inst();
    UINT mpirank = mpiutil.get_rank();
    UINT mpisize = mpiutil.get_size();
    ITYPE inner_dim = dim / mpisize;
    ITYPE offs = inner_dim * mpirank;

    Random random;
    random.set_seed(2022);

    QuantumState state_ref(n, false);
    QuantumState state(n, true);
    ComplexVector state_eigen(dim);

    state.set_Haar_random_state();
    state_ref.load(&state);
    for (ITYPE i = 0; i < dim; ++i) state_eigen[i] = state_ref.data_cpp()[i];

    QuantumCircuit circuit(n);
    UINT target, target_sub;
    double angle;
    std::complex<double> imag_unit(0, 1);

    target = random.int32() % n;
    circuit.add_X_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(
                      target, get_eigen_matrix_single_Pauli(1), n) *
                  state_eigen;

    target = random.int32() % n;
    circuit.add_Y_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(
                      target, get_eigen_matrix_single_Pauli(2), n) *
                  state_eigen;

    target = random.int32() % n;
    circuit.add_Z_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(
                      target, get_eigen_matrix_single_Pauli(3), n) *
                  state_eigen;

    target = random.int32() % n;
    circuit.add_H_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, H, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_S_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, S, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_Sdag_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, S.adjoint(), n) *
        state_eigen;

    target = random.int32() % n;
    circuit.add_T_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, T, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_Tdag_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, T.adjoint(), n) *
        state_eigen;

    target = random.int32() % n;
    circuit.add_sqrtX_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, sqrtX, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_sqrtXdag_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, sqrtX.adjoint(), n) *
        state_eigen;

    target = random.int32() % n;
    circuit.add_sqrtY_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, sqrtY, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_sqrtYdag_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, sqrtY.adjoint(), n) *
        state_eigen;

    target = random.int32() % n;
    circuit.add_P0_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, P0, n) * state_eigen;

    target = (target + 1) % n;
    circuit.add_P1_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, P1, n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RX_gate(target, angle);
    circuit.add_RotInvX_gate(target, angle);
    circuit.add_RotX_gate(target, angle);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target,
            cos(angle / 2) * Identity + imag_unit * sin(angle / 2) * X, n) *
        state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RY_gate(target, angle);
    circuit.add_RotInvY_gate(target, angle);
    circuit.add_RotY_gate(target, angle);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target,
            cos(angle / 2) * Identity + imag_unit * sin(angle / 2) * Y, n) *
        state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RZ_gate(target, angle);
    circuit.add_RotInvZ_gate(target, angle);
    circuit.add_RotZ_gate(target, angle);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target,
            cos(angle / 2) * Identity + imag_unit * sin(angle / 2) * Z, n) *
        state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_CNOT_gate(target, target_sub);
    state_eigen =
        get_eigen_matrix_full_qubit_CNOT(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_CZ_gate(target, target_sub);
    state_eigen =
        get_eigen_matrix_full_qubit_CZ(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_SWAP_gate(target, target_sub);
    state_eigen =
        get_eigen_matrix_full_qubit_SWAP(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_U1_gate(target, M_PI);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, Z, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_U2_gate(target, 0, M_PI);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, H, n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_U3_gate(target, -angle, 0, 0);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target,
            cos(angle / 2) * Identity + imag_unit * sin(angle / 2) * Y, n) *
        state_eigen;

    std::vector<UINT> target_index_list{0, 1, 2, 3};
    std::vector<UINT> pauli_id_list{0, 1, 2, 3};
    circuit.add_multi_Pauli_gate(target_index_list, pauli_id_list);

    // add same gate == cancel above pauli gate
    PauliOperator pauli = PauliOperator("I 0 X 1 Y 2 Z 3");
    circuit.add_multi_Pauli_gate(pauli);

    ComplexMatrix mat_x(2, 2);
    target = random.int32() % n;
    mat_x << 0, 1, 1, 0;
    circuit.add_dense_matrix_gate(target, mat_x);

    state_eigen = get_expanded_eigen_matrix_with_identity(
                      target, get_eigen_matrix_single_Pauli(1), n) *
                  state_eigen;

    circuit.update_quantum_state(&state);

    if (state.get_device_name() == "multi-cpu")
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state_eigen[i + offs] - state.data_cpp()[i]), 0, eps);
    else
        for (ITYPE i = 0; i < dim; ++i)
            ASSERT_NEAR(abs(state_eigen[i] - state.data_cpp()[i]), 0, eps);
}

TEST(CircuitTest_multicpu, CircuitRev) {
    const UINT n = 4;
    const UINT dim = 1ULL << n;
    MPIutil& mpiutil = MPIutil::get_inst();
    UINT mpirank = mpiutil.get_rank();
    UINT mpisize = mpiutil.get_size();
    ITYPE inner_dim = dim / mpisize;
    ITYPE offs = inner_dim * mpirank;

    Random random;
    random.set_seed(2022);

    QuantumState state(n, true);
    QuantumState state_ref(n, false);
    ComplexVector state_eigen(dim);

    state.set_Haar_random_state();
    state_ref.load(&state);
    for (ITYPE i = 0; i < dim; ++i) state_eigen[i] = state_ref.data_cpp()[i];

    QuantumCircuit circuit(n);
    UINT target, target_sub;
    double angle;

    target = random.int32() % n;
    circuit.add_X_gate(target);

    target = random.int32() % n;
    circuit.add_Y_gate(target);

    target = random.int32() % n;
    circuit.add_Z_gate(target);

    target = random.int32() % n;
    circuit.add_H_gate(target);

    target = random.int32() % n;
    circuit.add_S_gate(target);

    target = random.int32() % n;
    circuit.add_Sdag_gate(target);

    target = random.int32() % n;
    circuit.add_T_gate(target);

    target = random.int32() % n;
    circuit.add_Tdag_gate(target);

    target = random.int32() % n;
    circuit.add_sqrtX_gate(target);

    target = random.int32() % n;
    circuit.add_sqrtXdag_gate(target);

    target = random.int32() % n;
    circuit.add_sqrtY_gate(target);

    target = random.int32() % n;
    circuit.add_sqrtYdag_gate(target);

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RX_gate(target, angle);
    circuit.add_RotInvX_gate(target, angle);
    circuit.add_RotX_gate(target, angle);

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RY_gate(target, angle);
    circuit.add_RotInvY_gate(target, angle);
    circuit.add_RotInvY_gate(target, angle);

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RZ_gate(target, angle);
    circuit.add_RotInvZ_gate(target, angle);
    circuit.add_RotZ_gate(target, -angle);

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_CNOT_gate(target, target_sub);

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_CZ_gate(target, target_sub);

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_SWAP_gate(target, target_sub);

    Observable observable(n);
    angle = 2 * PI * random.uniform();

    observable.add_operator(1.0, "Z 0");
    observable.add_operator(2.0, "Z 0 Z 1");

    circuit.add_diagonal_observable_rotation_gate(observable, angle);

    circuit.update_quantum_state(&state);

    auto revcircuit = circuit.get_inverse();

    revcircuit->update_quantum_state(&state);

    if (state.get_device_name() == "multi-cpu")
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state_eigen[i + offs] - state.data_cpp()[i]), 0, eps);
    else
        for (ITYPE i = 0; i < dim; ++i)
            ASSERT_NEAR(abs(state_eigen[i] - state.data_cpp()[i]), 0, eps);

    delete revcircuit;
}

TEST(CircuitTest_multicpu, CircuitOptimize) {
    MPIutil& m = MPIutil::get_inst();
    const UINT num_global_qubit = (UINT)std::log2(m.get_size());
    UINT n = 4 + num_global_qubit;
    const UINT dim = 1ULL << n;

    {
        // merge successive gates
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(0);
        circuit.add_Y_gate(0);
        UINT block_size = 2;
        UINT expected_depth = 1;
        UINT expected_gate_count = 1;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // tensor product, merged
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(0);
        circuit.add_Y_gate(1);
        UINT block_size = 2;
        UINT expected_depth = 1;
        UINT expected_gate_count = 1;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // do not take tensor product
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(0);
        circuit.add_Y_gate(1);
        UINT block_size = 1;
        UINT expected_depth = 1;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, control does not commute with X
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(0);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_Y_gate(0);
        UINT block_size = 1;
        UINT expected_depth = 3;
        UINT expected_gate_count = 3;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, control does not commute with Z
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(0);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_Z_gate(0);
        UINT block_size = 1;
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, control commute with Z
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_Z_gate(0);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_Z_gate(0);
        UINT block_size = 1;
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(1);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_X_gate(1);
        UINT block_size = 1;
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_Z_gate(1);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_X_gate(1);
        UINT block_size = 1;
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(1);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_Z_gate(1);
        UINT block_size = 1;
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_Z_gate(1);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_Z_gate(1);
        UINT block_size = 1;
        UINT expected_depth = 3;
        UINT expected_gate_count = 3;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);

        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_Z_gate(1);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_Z_gate(1);
        UINT block_size = 2;
        UINT expected_depth = 1;
        UINT expected_gate_count = 1;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        auto gate1 = gate::CNOT(0, 1);
        auto gate2 = gate::CNOT(1, 2);
        auto gate3 = gate::Y(2);

        circuit.add_Z_gate(0);
        circuit.add_gate(gate::merge(gate1, gate3));
        circuit.add_gate(gate::merge(gate2, gate3));
        circuit.add_Z_gate(1);

        delete gate1;
        delete gate2;
        delete gate3;

        UINT block_size = 2;
        UINT expected_depth = 3;
        UINT expected_gate_count = 3;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        test_state.load(&state);
        QuantumCircuit circuit(n);

        auto gate1 = gate::CNOT(0, 1);
        auto gate2 = gate::CNOT(1, 2);
        auto gate3 = gate::Y(2);

        circuit.add_Z_gate(0);
        circuit.add_gate(gate::merge(gate1, gate3));
        circuit.add_gate(gate::merge(gate2, gate3));
        circuit.add_Z_gate(1);

        delete gate1;
        delete gate2;
        delete gate3;

        UINT block_size = 3;
        UINT expected_depth = 1;
        UINT expected_gate_count = 1;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        ASSERT_STATE_NEAR(state, test_state, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }
}

TEST(CircuitTest_multicpu, RandomCircuitOptimize) {
    const UINT n = 5;
    const UINT dim = 1ULL << n;
    const UINT depth = 5;
    Random random;
    random.set_seed(2022);

    UINT max_repeat = 3;
    // UINT max_block_size = n;
    UINT max_block_size = 1;  // TODO: multi-cpu limitation.

    for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
        QuantumState state(n, true), org_state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        org_state.load(&state);
        QuantumCircuit circuit(n);

        for (UINT d = 0; d < depth; ++d) {
            for (UINT i = 0; i < n; ++i) {
                UINT r = random.int32() % 5;
                if (r == 0)
                    circuit.add_sqrtX_gate(i);
                else if (r == 1)
                    circuit.add_sqrtY_gate(i);
                else if (r == 2)
                    circuit.add_T_gate(i);
                else if (r == 3) {
                    if (i + 1 < n) circuit.add_CNOT_gate(i, i + 1);
                } else if (r == 4) {
                    if (i + 1 < n) circuit.add_CZ_gate(i, i + 1);
                }
            }
        }

        test_state.load(&org_state);
        circuit.update_quantum_state(&test_state);
        QuantumCircuitOptimizer qco;
        for (UINT block_size = 1; block_size <= max_block_size; ++block_size) {
            QuantumCircuit* copy_circuit = circuit.copy();
            qco.optimize(copy_circuit, block_size);
            state.load(&org_state);
            copy_circuit->update_quantum_state(&state);
            ASSERT_STATE_NEAR(state, test_state, eps);
            delete copy_circuit;
        }
    }
}

TEST(CircuitTest_multicpu, RandomCircuitOptimize2) {
    const UINT n = 5;
    const UINT dim = 1ULL << n;
    const UINT depth = 10;
    Random random;
    random.set_seed(2022);

    UINT max_repeat = 3;
    // UINT max_block_size = n;
    UINT max_block_size = 1;  // TODO: multi-cpu limitation.

    for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
        QuantumState state(n, true), org_state(n, true), test_state(n, true);
        state.set_Haar_random_state();
        org_state.load(&state);
        QuantumCircuit circuit(n);

        for (UINT d = 0; d < depth; ++d) {
            for (UINT i = 0; i < n; ++i) {
                UINT r = random.int32() % 6;
                if (r == 0)
                    circuit.add_sqrtX_gate(i);
                else if (r == 1)
                    circuit.add_sqrtY_gate(i);
                else if (r == 2)
                    circuit.add_T_gate(i);
                else if (r == 3) {
                    UINT r2 = random.int32() % n;
                    if (r2 == i) r2 = (r2 + 1) % n;
                    if (i + 1 < n) circuit.add_CNOT_gate(i, r2);
                } else if (r == 4) {
                    UINT r2 = random.int32() % n;
                    if (r2 == i) r2 = (r2 + 1) % n;
                    if (i + 1 < n) circuit.add_CZ_gate(i, r2);
                } else if (r == 5) {
                    UINT r2 = random.int32() % n;
                    if (r2 == i) r2 = (r2 + 1) % n;
                    if (i + 1 < n) circuit.add_SWAP_gate(i, r2);
                }
            }
        }

        test_state.load(&org_state);
        circuit.update_quantum_state(&test_state);
        QuantumCircuitOptimizer qco;
        for (UINT block_size = 1; block_size <= max_block_size; ++block_size) {
            QuantumCircuit* copy_circuit = circuit.copy();
            qco.optimize(copy_circuit, block_size);
            state.load(&org_state);
            copy_circuit->update_quantum_state(&state);
            ASSERT_STATE_NEAR(state, test_state, eps);
            delete copy_circuit;
        }
    }
}

TEST(CircuitTest_multicpu, RandomCircuitOptimize3) {
    const UINT n = 5;
    const UINT dim = 1ULL << n;
    const UINT depth = 10 * n;
    Random random;
    random.set_seed(2022);
    MPIutil& mpiutil = MPIutil::get_inst();
    UINT mpirank = mpiutil.get_rank();
    UINT mpisize = mpiutil.get_size();

    UINT max_repeat = 3;
    // UINT max_block_size = n;
    UINT max_block_size = 1;  // TODO: multi-cpu limitation.
    UINT seed = 2021;
    std::mt19937 eng(seed);

    std::vector<UINT> qubit_list;
    for (int i = 0; i < n; ++i) qubit_list.push_back(i);

    for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
        QuantumState state(n, true), org_state(n, true), test_state(n, true);
        state.set_Haar_random_state(2000);
        org_state.load(&state);
        QuantumCircuit circuit(n);

        for (UINT d = 0; d < depth; ++d) {
            std::shuffle(qubit_list.begin(), qubit_list.end(), eng);
            std::vector<UINT> mylist;
            mylist.push_back(qubit_list[0]);
            mylist.push_back(qubit_list[1]);
            circuit.add_random_unitary_gate(mylist, seed + d);
            // circuit.add_CNOT_gate(qubit_list[0], qubit_list[1]);
        }

        test_state.load(&org_state);
        circuit.update_quantum_state(&test_state);
        // std::cout << "#original-circuit " << mpirank << ":" << circuit <<
        // std::endl; std::cout << "# test_state" << mpirank << ": " <<
        // test_state << std::endl;
        QuantumCircuitOptimizer qco;
        for (UINT block_size = 1; block_size <= max_block_size; ++block_size) {
            QuantumCircuit* copy_circuit = circuit.copy();
            qco.optimize(copy_circuit, block_size);
            state.load(&org_state);
            copy_circuit->update_quantum_state(&state);
            // std::cout << "#optimized-circuit " << mpirank << ":" <<
            // block_size << ", " << circuit << std::endl; std::cout << "#
            // state" << mpirank << ": " << state << std::endl;
            ASSERT_STATE_NEAR(state, test_state, eps);
            delete copy_circuit;
        }
    }
}

TEST(CircuitTest_multicpu, SuzukiTrotterExpansion) {
    CPPCTYPE J(0.0, 1.0);
    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();

    const UINT n = 2;
    const UINT dim = 1ULL << n;

    double angle;
    std::vector<double> coef;

    const UINT seed = 1918;
    Random random;
    random.set_seed(seed);

    CPPCTYPE res;
    CPPCTYPE test_res;

    Observable diag_observable(n), non_diag_observable(n), observable(n);
    Eigen::MatrixXcd test_observable;

    QuantumState state(n, true);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);

    QuantumCircuit circuit(n);
    Eigen::MatrixXcd test_circuit;

    for (ITYPE i = 0; i < 6; ++i) {
        coef.push_back(-random.uniform());
        // coef.push_back(-1.);
    }
    angle = 2 * PI * random.uniform();

    observable.add_operator(coef[0], "Z 0 I 1");
    observable.add_operator(coef[1], "X 0 Y 1");
    observable.add_operator(coef[2], "Z 0 Z 1");
    observable.add_operator(coef[3], "Z 0 X 1");
    observable.add_operator(coef[4], "Y 0 X 1");
    observable.add_operator(coef[5], "I 0 Z 1");

    test_observable =
        coef[0] * get_expanded_eigen_matrix_with_identity(0, Z, n);
    test_observable += coef[1] * kronecker_product(Y, X);
    test_observable += coef[2] * kronecker_product(Z, Z);
    test_observable += coef[3] * kronecker_product(X, Z);
    test_observable += coef[4] * kronecker_product(X, Y);
    test_observable +=
        coef[5] * get_expanded_eigen_matrix_with_identity(1, Z, n);

    const auto num_repeats =
        static_cast<UINT>(std::ceil(angle * (double)n * 100.));
    // circuit.add_diagonal_observable_rotation_gate(diag_observable, angle);
    circuit.add_observable_rotation_gate(observable, angle, num_repeats);

    test_circuit = J * angle * test_observable;
    test_circuit = test_circuit.exp();

    state.set_computational_basis(0);
    test_state(0) = 1.;

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state)(0, 0);

    circuit.update_quantum_state(&state);
    test_state = test_circuit * test_state;

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state)(0, 0);
    ASSERT_NEAR(abs(test_res.real() - res.real()) / test_res.real(), 0, 0.01);

    state.set_Haar_random_state(seed);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state.data_cpp()[i];

    test_state = test_circuit * test_state;
    circuit.update_quantum_state(&state);

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state)(0, 0);
    ASSERT_NEAR(abs(test_res.real() - res.real()) / test_res.real(), 0, 0.01);
}

TEST(CircuitTest_multicpu, RotateDiagonalObservable) {
    CPPCTYPE J(0.0, 1.0);
    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();

    const UINT n = 2;
    const UINT dim = 1ULL << n;

    double angle, coef1, coef2;
    Random random;
    random.set_seed(2022);

    CPPCTYPE res;
    CPPCTYPE test_res;

    Observable observable(n);
    Eigen::MatrixXcd test_observable;

    QuantumState state(n, true);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);

    QuantumCircuit circuit(n);
    Eigen::MatrixXcd test_circuit;

    coef1 = -random.uniform();
    coef2 = -random.uniform();
    angle = 2 * PI * random.uniform();

    observable.add_operator(coef1, "Z 0");
    observable.add_operator(coef2, "Z 0 Z 1");

    test_observable = coef1 * get_expanded_eigen_matrix_with_identity(0, Z, n);
    test_observable += coef2 * kronecker_product(Z, Z);

    circuit.add_diagonal_observable_rotation_gate(observable, angle);
    test_circuit = (J * angle * test_observable).exp();

    state.set_computational_basis(0);
    test_state(0) = 1.;

    circuit.update_quantum_state(&state);
    test_state = test_circuit * test_state;

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state)(0, 0);

    ASSERT_NEAR(res.imag(), 0, eps);
    ASSERT_NEAR(test_res.imag(), 0, eps);

    state.set_Haar_random_state();
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state.data_cpp()[i];

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state)(0, 0);

    test_state = test_circuit * test_state;
    circuit.update_quantum_state(&state);

    res = observable.get_expectation_value(&state);
    test_res = (test_state.adjoint() * test_observable * test_state)(0, 0);

    ASSERT_NEAR(abs(test_res.real() - res.real()) / test_res.real(), 0, 0.01);
    ASSERT_NEAR(res.imag(), 0, eps);
    ASSERT_NEAR(test_res.imag(), 0, eps);
}

TEST(CircuitTest_multicpu, SpecialGatesToString) {
    QuantumState state(1, true);
    QuantumCircuit c(1);
    c.add_gate(gate::DepolarizingNoise(0, 0));
    c.update_quantum_state(&state);
    std::string s = c.to_string();
}

TEST(CircuitTest_multicpu, MergeCircuits) {
    QuantumState state(2, true);
    QuantumCircuit circuit1(2), circuit2(2);
    circuit1.add_X_gate(0);
    circuit2.add_X_gate(1);
    state.set_zero_state();
    circuit1.merge_circuit(&circuit2);
    circuit1.update_quantum_state(&state);
    ASSERT_NEAR(abs(state.data_cpp()[3]), 1.0, 0.0001);
}

void _PrintCircuit(QuantumCircuit* circuit) {
    for (auto& gate : circuit->gate_list) {
        auto t_index_list = gate->get_target_index_list();
        auto c_index_list = gate->get_control_index_list();
        std::cout << gate->get_name() << "(t:{";
        for (auto idx : t_index_list) {
            std::cout << idx << ",";
        }
        std::cout << "}, c:{";
        for (auto idx : c_index_list) {
            std::cout << idx << ",";
        }
        std::cout << "})" << std::endl;
    }
}

void _ApplyOptimizer(QuantumCircuit* circuit_ref, int opt_lv, UINT swap_lv,
    UINT num_exp_outer_swaps) {
    const UINT n = circuit_ref->qubit_count;
    const ITYPE dim = 1ULL << n;

    QuantumState state_ref(n);
    QuantumState state(n, true);

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = inner_dim * m.get_rank();

    {
        state_ref.set_Haar_random_state(2022);
        for (ITYPE i = 0; i < inner_dim; ++i)
            state.data_cpp()[i] = state_ref.data_cpp()[(i + offs) % dim];

        QuantumCircuit* circuit = circuit_ref->copy();
        QuantumCircuitOptimizer qco;
        if (opt_lv >= 0) {
            qco.optimize(circuit, opt_lv, swap_lv);
        } else {
            qco.optimize_light(circuit, swap_lv);
        }

        circuit->update_quantum_state(&state);
        circuit_ref->update_quantum_state(&state_ref);

#if 0
        if (m.get_rank() == 0) {
            std::cout << "-----original circuit-----" << std::endl;
            _PrintCircuit(circuit_ref);
            std::cout << "-----optimized circuit-----" << std::endl;
            _PrintCircuit(circuit);
            std::cout << "---------------------------" << std::endl;
        }
#endif

        // check if all target qubits are inner except for SWAP, FusedSWAP,
        // and diagonal gate (e.g., CZ, RZ)
        for (auto& gate : circuit->gate_list) {
            auto gate_name = gate->get_name();
            if (gate_name == "SWAP" || gate_name == "FusedSWAP" ||
                gate_name == "I" || gate_name == "Z" ||
                gate_name == "Z-rotation" || gate_name == "CZ" ||
                gate_name == "Projection-0" || gate_name == "Projection-1" ||
                gate_name == "S" || gate_name == "Sdag" || gate_name == "T" ||
                gate_name == "Tdag" || gate_name == "DiagonalMatrix") {
                continue;
            }
            auto t_index_list = gate->get_target_index_list();
            for (auto idx : t_index_list) {
                ASSERT_TRUE(idx < state.inner_qc);
            }
        }

        // check the number of SWAP and FusedSWAP is the same with expected.
        UINT num_outer_swap_gates = 0;
        for (auto& gate : circuit->gate_list) {
            if (gate->get_name() == "SWAP" || gate->get_name() == "FusedSWAP") {
                auto t_index_list = gate->get_target_index_list();
                for (auto idx : t_index_list) {
                    if (idx >= state.inner_qc) {
                        num_outer_swap_gates++;
                        break;
                    }
                }
            }
        }

#if 0
        if (m.get_rank() == 0) {
            std::cout << "num_outer_swap_gates=" << num_outer_swap_gates << std::endl;
            std::cout << "num_exp_outer_swap_gates=" << num_exp_outer_swaps << std::endl;
        }
#endif
        ASSERT_TRUE(num_outer_swap_gates <= num_exp_outer_swaps)
            << "# outer swaps is " << num_outer_swap_gates
            << ", but expected is " << num_exp_outer_swaps;

        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(abs(state.data_cpp()[i] -
                            state_ref.data_cpp()[(i + offs) % dim]),
                0, eps)
                << "[rank:" << m.get_rank() << "] Optimizer diff at " << i;

        delete circuit;
    }
}

TEST(CircuitTest_multicpu, FSWAPOptimizer_6qubits) {
    UINT n = 6;

    MPIutil& m = MPIutil::get_inst();
    const UINT log_nodes = std::log2(m.get_size());
    if (log_nodes == 0 || log_nodes > n - 2) {
        return;
    }
    const UINT outer_qc = log_nodes;
    const UINT inner_qc = n - log_nodes;

    Random random;
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        circuit.add_RX_gate(0, random.uniform() * 3.14159);
        circuit.add_RX_gate(n - 1, random.uniform() * 3.14159);
        _ApplyOptimizer(&circuit, 0, 1, 2);
    }

    if (inner_qc >= outer_qc * 2) {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT rep = 0; rep < 2; rep++) {
            for (UINT i = 0; i < n; i++) {
                circuit.add_H_gate(i);
            }
        }
        // TODO gate順序変更に対応したら2回に変更
        _ApplyOptimizer(&circuit, 0, 1, 4);
    }

    if (inner_qc >= outer_qc * 2) {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT rep = 0; rep < 2; rep++) {
            for (UINT i = 0; i < n; i++) {
                circuit.add_CNOT_gate(i, (i + 1) % n);
            }
        }
        // TODO gate順序変更に対応したら2回に変更
        _ApplyOptimizer(&circuit, 0, 1, 4);
    }

    if (outer_qc <= n / 2) {
        random.set_seed(2022);
        QuantumCircuit circuit(n);

        // outer
        for (UINT rep = 0; rep < n * 2; rep++) {
            circuit.add_RX_gate(
                inner_qc + (rep % outer_qc), random.uniform() * 3.14159);
        }
        // inner
        for (UINT rep = 0; rep < n * 2; rep++) {
            circuit.add_RX_gate((rep % inner_qc), random.uniform() * 3.14159);
        }
        // outer
        for (UINT rep = 0; rep < n * 2; rep++) {
            circuit.add_RX_gate(
                inner_qc + (rep % outer_qc), random.uniform() * 3.14159);
        }

        _ApplyOptimizer(&circuit, 0, 1, 4);
    }

    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);

        for (UINT rep = 0; rep < n * 4; rep++) {
            circuit.add_RX_gate(
                ((UINT)(random.uniform() * n)) % n, random.uniform() * 3.14159);
        }

        _ApplyOptimizer(&circuit, 0, 1, n * 8);
    }

    if (inner_qc >= 3 && outer_qc >= 3) {
        random.set_seed(2022);
        QuantumCircuit circuit(n);

        circuit.add_H_gate(n - 1);
        circuit.add_H_gate(n - 3);

        _ApplyOptimizer(&circuit, 0, 1, 2);
    }
}

TEST(CircuitTest_multicpu, FSWAPOptimizer_nocomm_6qubits) {
    UINT n = 6;

    MPIutil& m = MPIutil::get_inst();
    const UINT log_nodes = std::log2(m.get_size());
    if (log_nodes == 0 || log_nodes > n - 2) {
        return;
    }
    const UINT outer_qc = log_nodes;
    const UINT inner_qc = n - log_nodes;

    Random random;

    std::vector<int> block_sizes = {
        0, (int)n, -1};  // -1 means using optimize_light
    const UINT swap_lv = 1;
    const UINT depth = 2;

    for (auto block_size : block_sizes) {
        // X, Yゲートはouter qubitを使わなければFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < inner_qc; i++) {
                    circuit.add_X_gate(i);
                    circuit.add_Y_gate(i);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // CNOTゲートはtargetでouter qubitを使わなければFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < n; i++) {
                    if ((i + 1) % n < inner_qc) {
                        circuit.add_CNOT_gate(i, (i + 1) % n);
                    }
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // Hゲートはouter qubitを使わなければFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < inner_qc; i++) {
                    circuit.add_H_gate(i);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // RX, RYゲートはouter qubitを使わなければFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < inner_qc; i++) {
                    circuit.add_RX_gate(i, random.uniform() * 3.14159);
                    circuit.add_RY_gate(i, random.uniform() * 3.14159);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // SqrtX, SqrtXdag, SqrtY, SqrtYdagゲートはtargetでouter
        // qubitを使わなければFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < inner_qc; i++) {
                    circuit.add_sqrtX_gate(i);
                    circuit.add_sqrtXdag_gate(i);
                    circuit.add_sqrtY_gate(i);
                    circuit.add_sqrtYdag_gate(i);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // U1, U2, U3ゲートはtargetでouter qubitを使わなければFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < inner_qc; i++) {
                    circuit.add_U1_gate(i, random.uniform() * 3.14159);
                    circuit.add_U2_gate(i, random.uniform() * 3.14159,
                        random.uniform() * 3.14159);
                    circuit.add_U3_gate(i, random.uniform() * 3.14159,
                        random.uniform() * 3.14159, random.uniform() * 3.14159);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // DenseMatrixゲートはtargetでouter qubitを使わなければFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < inner_qc; i++) {
                    std::vector<UINT> target{i};
                    ComplexMatrix mat =
                        get_eigen_matrix_random_single_qubit_unitary();
                    auto DenseMatrix_gate = gate::DenseMatrix(target, mat);
                    circuit.add_gate(DenseMatrix_gate);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // Zゲートは全てのパターンでFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < n; i++) {
                    circuit.add_Z_gate(i);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // CZゲートは全てのパターンでFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < n; i++) {
                    circuit.add_CZ_gate(i, (i + 1) % n);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // Iゲートは全てのパターンでFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < n; i++) {
                    auto I_gate = gate::Identity(i);
                    circuit.add_gate(I_gate);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // P0, P1ゲートは全てのパターンでFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < n; i++) {
                    circuit.add_P0_gate(i);
                    circuit.add_P1_gate(i);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // RZゲートは全てのパターンでFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < n; i++) {
                    circuit.add_RZ_gate(i, random.uniform() * 3.14159);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // S, Sdag, T, Tdagゲートは全てのパターンでFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < n; i++) {
                    circuit.add_S_gate(i);
                    circuit.add_Sdag_gate(i);
                    circuit.add_T_gate(i);
                    circuit.add_Tdag_gate(i);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
        // DiagonalMatrixゲートは全てのパターンでFSWAP不要
        {
            random.set_seed(2022);
            QuantumCircuit circuit(n);
            for (UINT d = 0; d < depth; d++) {
                for (UINT i = 0; i < n; i++) {
                    std::vector<UINT> target{i};
                    ComplexVector diag =
                        get_eigen_diagonal_matrix_random_multi_qubit_unitary(1);
                    auto DiagonalMatrix_gate =
                        gate::DiagonalMatrix(target, diag);
                    circuit.add_gate(DiagonalMatrix_gate);
                }
            }
            _ApplyOptimizer(&circuit, block_size, swap_lv, 0);
        }
    }
}

void _BuildQulacsBenchmark(
    QuantumCircuit& circuit, UINT nqubits, UINT depth, Random& random) {
    // first rotation
    for (UINT k = 0; k < nqubits; k++) {
        circuit.add_RX_gate(k, random.uniform() * 3.14159);
        circuit.add_RZ_gate(k, random.uniform() * 3.14159);
    }

    // entangler
    for (UINT k = 0; k < nqubits; k++) {
        circuit.add_CNOT_gate(k, (k + 1) % nqubits);
    }

    for (UINT i = 0; i < depth; i++) {
        // mid rotation
        for (UINT k = 0; k < nqubits; k++) {
            circuit.add_RZ_gate(k, random.uniform() * 3.14159);
            circuit.add_RX_gate(k, random.uniform() * 3.14159);
            circuit.add_RZ_gate(k, random.uniform() * 3.14159);
        }

        // entangler
        for (UINT k = 0; k < nqubits; k++) {
            circuit.add_CNOT_gate(k, (k + 1) % nqubits);
        }
    }

    // last rotation
    for (UINT k = 0; k < nqubits; k++) {
        circuit.add_RZ_gate(k, random.uniform() * 3.14159);
        circuit.add_RX_gate(k, random.uniform() * 3.14159);
    }
}

void _BuildQVolumeBenchmark(
    QuantumCircuit& circuit, UINT nqubits, UINT depth, Random& random) {
    for (UINT d = 0; d < depth; ++d) {
        std::vector<UINT> index(nqubits), perm;
        std::iota(index.begin(), index.end(), 0);  // [0, ..., nqubits-1]
        for (UINT i = 0; i < nqubits; i++) {
            auto p = index.begin() + random.int32() % index.size();
            perm.push_back(*p);
            index.erase(p);
        }

        for (UINT w = 0; w < nqubits; w += 2) {
            std::vector<UINT> target_qubits = {perm[w], perm[w + 1]};
            circuit.add_random_unitary_gate(target_qubits, random.int32());
        }
    }
}

TEST(CircuitTest_multicpu, FSWAPOptimizer_reorder_6qubits) {
    UINT n = 6;

    MPIutil& m = MPIutil::get_inst();
    const UINT log_nodes = std::log2(m.get_size());
    if (log_nodes == 0 || log_nodes > n - 2) {
        return;
    }
    const UINT outer_qc = log_nodes;
    const UINT inner_qc = n - log_nodes;

    Random random;

    std::vector<int> block_sizes = {0, (int)n, -1};

    // Qulacs Benchmark
    for (int block_size : block_sizes) {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        _BuildQulacsBenchmark(circuit, n, 9, random);
        UINT n_expected_swaps = 0;
        switch (outer_qc) {
            case 1:
                n_expected_swaps = 14;
                break;
            case 2:
                n_expected_swaps = 18;
                break;
            case 3:
                n_expected_swaps = 23;
                break;
            case 4:
                n_expected_swaps = 68;
                break;
        }
        _ApplyOptimizer(&circuit, block_size, 2, n_expected_swaps);
    }

    // QVolume Benchmark
    for (int block_size : block_sizes) {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        _BuildQVolumeBenchmark(circuit, n, 10, random);
        UINT n_expected_swaps = 0;
        switch (outer_qc) {
            case 1:
                n_expected_swaps = 6;
                break;
            case 2:
                n_expected_swaps = 9;
                break;
            case 3:
                n_expected_swaps = 21;
                break;
            case 4:
                n_expected_swaps = 38;
                break;
        }
        _ApplyOptimizer(&circuit, 0, 2, n_expected_swaps);
    }

    // check for a reorder bug
    if (outer_qc == 1) {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < n; i++) {
            circuit.add_RX_gate(i, random.uniform() * 3.14159);
        }
        {
            QuantumGateBase* merged_gates =
                gate::RY(0, random.uniform() * 3.14159);
            for (UINT i = 1; i + 1 < n; i++) {
                merged_gates = gate::merge(
                    merged_gates, gate::RY(i, random.uniform() * 3.14159));
            }
            circuit.add_gate(merged_gates);
        }
        circuit.add_RY_gate(n - 1, random.uniform() * 3.14159);
        {
            QuantumGateBase* merged_gates =
                gate::RZ(0, random.uniform() * 3.14159);
            for (UINT i = 1; i + 1 < n; i++) {
                merged_gates = gate::merge(
                    merged_gates, gate::RZ(i, random.uniform() * 3.14159));
            }
            circuit.add_gate(merged_gates);
        }

        _ApplyOptimizer(&circuit, 0, 2, 2);
    }
}

TEST(CircuitTest_multicpu, FSWAPOptimizer_global_control) {
    UINT n = 6;

    MPIutil& m = MPIutil::get_inst();
    const UINT log_nodes = std::log2(m.get_size());
    if (log_nodes == 0 || log_nodes > n - 2) {
        return;
    }
    const UINT outer_qc = log_nodes;
    const UINT inner_qc = n - log_nodes;

    {
        QuantumCircuit circuit(n);
        circuit.add_X_gate(0);
        circuit.add_CNOT_gate(n - 1, 0);

        // expect that gates are not merged
        _ApplyOptimizer(&circuit, -1, 2, 0);
    }

    {
        QuantumCircuit circuit(n);
        // TOFFOLI(n-1, 1, 0)
        auto ptr = gate::X(0);
        auto toffoli = gate::to_matrix_gate(ptr);
        toffoli->add_control_qubit(n - 1, 1);
        toffoli->add_control_qubit(1, 1);
        delete ptr;
        circuit.add_gate(toffoli);
        // CNOT(n-1, 0)
        circuit.add_CNOT_gate(n - 1, 0);

        // expect that gates are merged to
        // DenseMatrix(target=[0, 1], control=[n-1])
        _ApplyOptimizer(&circuit, -1, 2, 0);
    }

    {
        QuantumCircuit circuit(n);
        // TOFFOLI(n-1, 1, 0)
        auto ptr = gate::X(0);
        auto toffoli = gate::to_matrix_gate(ptr);
        toffoli->add_control_qubit(n - 1, 1);
        toffoli->add_control_qubit(1, 1);
        delete ptr;
        circuit.add_gate(toffoli);
        // CNOT(1, 0)
        circuit.add_CNOT_gate(1, 0);

        // expect that gates are not merged
        _ApplyOptimizer(&circuit, -1, 2, 0);
    }
}

TEST(CircuitTest_multicpu, FSWAPOptimizer_with_mpisize) {
    UINT n = 6;

    QuantumCircuit circuit(n);
    for (UINT k = 0; k < n; k++) {
        circuit.add_H_gate(k);
        circuit.add_X_gate(k);
    }

    QuantumCircuitOptimizer qco(4);
    qco.optimize(&circuit, 2, 2);

    bool has_swap_gate = false;
    for (auto& gate : circuit.gate_list) {
        auto gate_name = gate->get_name();
        if (gate_name == "FusedSWAP") {
            has_swap_gate = true;
        }
    }
    ASSERT_TRUE(has_swap_gate)
        << "circuit optimized with mpi_size=4 should have FusedSWAP gates";
}
#endif
