#ifdef _USE_MPI
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cppsim/gate.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/utility.hpp>
#include <functional>
#include <iostream>

#include "../util/util.hpp"

TEST(GateTest_multicpu, ApplySingleQubitGate) {
    Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2), H(2, 2),
        S(2, 2), T(2, 2), sqrtX(2, 2), sqrtY(2, 2), P0(2, 2), P1(2, 2);

    Identity << 1, 0, 0, 1;
    X << 0, 1, 1, 0;
    Y << 0, -1.i, 1.i, 0;
    Z << 1, 0, 0, -1;
    H << 1, 1, 1, -1;
    H /= sqrt(2.);
    S << 1, 0, 0, 1.i;
    T << 1, 0, 0, (1. + 1.i) / sqrt(2.);
    sqrtX << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
    sqrtY << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    const UINT n = 8;
    const ITYPE dim = 1ULL << n;

    Random random;
    random.set_seed(2022);
    QuantumState state_ref(n);
    QuantumState state(n, 1);
    std::vector<
        std::pair<std::function<QuantumGateBase*(UINT)>, Eigen::MatrixXcd>>
        funclist;
    funclist.push_back(std::make_pair(gate::Identity, Identity));
    funclist.push_back(std::make_pair(gate::X, X));
    funclist.push_back(std::make_pair(gate::Y, Y));
    funclist.push_back(std::make_pair(gate::Z, Z));
    funclist.push_back(std::make_pair(gate::H, H));
    funclist.push_back(std::make_pair(gate::S, S));
    funclist.push_back(std::make_pair(gate::Sdag, S.adjoint()));
    funclist.push_back(std::make_pair(gate::T, T));
    funclist.push_back(std::make_pair(gate::Tdag, T.adjoint()));
    funclist.push_back(std::make_pair(gate::sqrtX, sqrtX));
    funclist.push_back(std::make_pair(gate::sqrtXdag, sqrtX.adjoint()));
    funclist.push_back(std::make_pair(gate::sqrtY, sqrtY));
    funclist.push_back(std::make_pair(gate::sqrtYdag, sqrtY.adjoint()));
    funclist.push_back(std::make_pair(gate::P0, P0));
    funclist.push_back(std::make_pair(gate::P1, P1));

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = inner_dim * m.get_rank();
    Eigen::VectorXcd test_state1 = Eigen::VectorXcd::Zero(dim);
    Eigen::VectorXcd test_state2 = Eigen::VectorXcd::Zero(dim);
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        // UINT j = 0;
        for (auto func_mat : funclist) {
            auto func = func_mat.first;
            auto mat = func_mat.second;
            UINT target = random.int32() % n;

            state_ref.set_Haar_random_state(2022);
            for (ITYPE i = 0; i < dim; ++i)
                test_state1[i] = state_ref.data_cpp()[i];
            for (ITYPE i = 0; i < dim; ++i)
                test_state2[i] = state_ref.data_cpp()[i];
            for (ITYPE i = 0; i < inner_dim; ++i)
                state.data_cpp()[i] = state_ref.data_cpp()[i + offs];

            auto gate = func(target);
            gate->update_quantum_state(&state);
            ComplexMatrix small_mat;
            gate->set_matrix(small_mat);
            test_state1 =
                get_expanded_eigen_matrix_with_identity(target, small_mat, n) *
                test_state1;
            test_state2 =
                get_expanded_eigen_matrix_with_identity(target, mat, n) *
                test_state2;

            for (ITYPE i = 0; i < inner_dim; ++i)
                ASSERT_NEAR(
                    abs(state.data_cpp()[i] - test_state1[i + offs]), 0, eps);
            for (ITYPE i = 0; i < inner_dim; ++i)
                ASSERT_NEAR(
                    abs(state.data_cpp()[i] - test_state2[i + offs]), 0, eps);
        }
    }
}

TEST(GateTest_multicpu, ApplySingleQubitRotationGate) {
    Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);

    Identity << 1, 0, 0, 1;
    X << 0, 1, 1, 0;
    Y << 0, -1.i, 1.i, 0;
    Z << 1, 0, 0, -1;

    const UINT n = 8;
    const ITYPE dim = 1ULL << n;

    Random random;
    random.set_seed(2022);
    QuantumState state_ref(n);
    QuantumState state(n, 1);
    std::vector<std::pair<std::function<QuantumGateBase*(UINT, double)>,
        Eigen::MatrixXcd>>
        funclist;
    funclist.push_back(std::make_pair(gate::RX, X));
    funclist.push_back(std::make_pair(gate::RY, Y));
    funclist.push_back(std::make_pair(gate::RZ, Z));

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = inner_dim * m.get_rank();
    Eigen::VectorXcd test_state1 = Eigen::VectorXcd::Zero(dim);
    Eigen::VectorXcd test_state2 = Eigen::VectorXcd::Zero(dim);
    for (UINT target = 0; target < n; ++target) {
        // UINT funcid=0;
        for (auto func_mat : funclist) {
            double angle = random.uniform() * 3.14159;

            auto func = func_mat.first;
            auto mat = cos(angle / 2) * Eigen::MatrixXcd::Identity(2, 2) +
                       1.i * sin(angle / 2) * func_mat.second;

            state_ref.set_Haar_random_state(2022);
            for (ITYPE i = 0; i < dim; ++i)
                test_state1[i] = state_ref.data_cpp()[i];
            for (ITYPE i = 0; i < dim; ++i)
                test_state2[i] = state_ref.data_cpp()[i];
            for (ITYPE i = 0; i < inner_dim; ++i)
                state.data_cpp()[i] = state_ref.data_cpp()[i + offs];

            auto gate = func(target, angle);
            gate->update_quantum_state(&state);
            ComplexMatrix small_mat;
            gate->set_matrix(small_mat);
            test_state1 =
                get_expanded_eigen_matrix_with_identity(target, small_mat, n) *
                test_state1;
            test_state2 =
                get_expanded_eigen_matrix_with_identity(target, mat, n) *
                test_state2;

            for (ITYPE i = 0; i < inner_dim; ++i)
                ASSERT_NEAR(
                    real(state.data_cpp()[i] - test_state1[i + offs]), 0, eps);
            for (ITYPE i = 0; i < inner_dim; ++i)
                ASSERT_NEAR(
                    real(state.data_cpp()[i] - test_state2[i + offs]), 0, eps);
            for (ITYPE i = 0; i < inner_dim; ++i)
                ASSERT_NEAR(
                    imag(state.data_cpp()[i] - test_state1[i + offs]), 0, eps);
            for (ITYPE i = 0; i < inner_dim; ++i)
                ASSERT_NEAR(
                    imag(state.data_cpp()[i] - test_state2[i + offs]), 0, eps);
        }
    }
}

TEST(GateTest_multicpu, SingleQubitUnitaryGate) {
    UINT n = 10;
    const ITYPE dim = 1ULL << n;

    QuantumState state_ref(n);
    QuantumState state(n, 1);

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = inner_dim * m.get_rank();

    Random random;
    random.set_seed(2022);
    for (UINT target = 0; target < n; ++target) {
        state_ref.set_Haar_random_state(2022);
        state.load(&state_ref);

        // update state
        double angle = random.uniform() * 3.14159;
        auto gate = gate::U1(target, angle);
        gate->update_quantum_state(&state);
        gate->update_quantum_state(&state_ref);

        for (ITYPE i = 0; i < inner_dim; ++i) {
            ASSERT_NEAR(real(state.data_cpp()[i]),
                real(state_ref.data_cpp()[(i + offs) % dim]), eps);
            ASSERT_NEAR(imag(state.data_cpp()[i]),
                imag(state_ref.data_cpp()[(i + offs) % dim]), eps);
        }
    }

    for (UINT target = 0; target < n; ++target) {
        state_ref.set_Haar_random_state(2022);
        state.load(&state_ref);

        // update state
        double angle1 = random.uniform() * 3.14159;
        double angle2 = random.uniform() * 3.14159;
        auto gate = gate::U2(target, angle1, angle2);
        gate->update_quantum_state(&state);
        gate->update_quantum_state(&state_ref);

        for (ITYPE i = 0; i < inner_dim; ++i) {
            ASSERT_NEAR(real(state.data_cpp()[i]),
                real(state_ref.data_cpp()[(i + offs) % dim]), eps);
            ASSERT_NEAR(imag(state.data_cpp()[i]),
                imag(state_ref.data_cpp()[(i + offs) % dim]), eps);
        }
    }

    for (UINT target = 0; target < n; ++target) {
        state_ref.set_Haar_random_state(2022);
        state.load(&state_ref);

        // update state
        double angle1 = random.uniform() * 3.14159;
        double angle2 = random.uniform() * 3.14159;
        double angle3 = random.uniform() * 3.14159;
        auto gate = gate::U3(target, angle1, angle2, angle3);
        gate->update_quantum_state(&state);
        gate->update_quantum_state(&state_ref);

        for (ITYPE i = 0; i < inner_dim; ++i) {
            ASSERT_NEAR(real(state.data_cpp()[i]),
                real(state_ref.data_cpp()[(i + offs) % dim]), eps);
            ASSERT_NEAR(imag(state.data_cpp()[i]),
                imag(state_ref.data_cpp()[(i + offs) % dim]), eps);
        }
    }
}

TEST(GateTest_multicpu, SingleQubitDiagonalGate) {
    UINT n = 10;
    const ITYPE dim = 1ULL << n;

    QuantumState state_ref(n);
    QuantumState state(n, 1);

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = inner_dim * m.get_rank();

    Random random;
    random.set_seed(2022);
    for (UINT target = 0; target < n; ++target) {
        state_ref.set_Haar_random_state(2022);
        state.load(&state_ref);

        // update state
        ComplexVector diag =
            get_eigen_diagonal_matrix_random_multi_qubit_unitary(1);
        auto gate = gate::DiagonalMatrix(std::vector<UINT>{target}, diag);
        gate->update_quantum_state(&state);
        gate->update_quantum_state(&state_ref);

        for (ITYPE i = 0; i < inner_dim; ++i) {
            ASSERT_NEAR(real(state.data_cpp()[i]),
                real(state_ref.data_cpp()[(i + offs) % dim]), eps);
            ASSERT_NEAR(imag(state.data_cpp()[i]),
                imag(state_ref.data_cpp()[(i + offs) % dim]), eps);
        }
    }
}

#if 0  // need update_quantum_seed with seed API or singletonize Random Class
TEST(GateTest_multicpu, MeasurementGate) {
    UINT n = 8;
    const ITYPE dim = 1ULL << n;

    QuantumState state_ref(n);
    QuantumState state(n, 1);

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = inner_dim * m.get_rank();

    for (UINT target = 0; target < n; ++target) {
        for (UINT classical = 0; classical < n; ++classical) {
                        UINT seed = target << 3 + classical + 3;
            if (target == classical) continue;
            state_ref.set_Haar_random_state(2022);
            state.load(&state_ref);

            for (ITYPE i = 0; i < inner_dim; ++i) {
                ASSERT_NEAR(real(state.data_cpp()[i]),
                    real(state_ref.data_cpp()[(i + offs) % dim]), eps)
                    << "rank: " << m.get_rank() << ", " << target << ", "
                    << classical;
                ASSERT_NEAR(imag(state.data_cpp()[i]),
                    imag(state_ref.data_cpp()[(i + offs) % dim]), eps)
                    << "rank: " << m.get_rank() << ", " << target << ", "
                    << classical;
            }

            // update state
            auto measurement = gate::Measurement(target, classical);
            measurement->update_quantum_state(&state, seed);
            measurement->update_quantum_state(&state_ref, seed);

            for (ITYPE i = 0; i < inner_dim; ++i) {
                ASSERT_NEAR(real(state.data_cpp()[i]),
                    real(state_ref.data_cpp()[(i + offs) % dim]), eps)
                    << "rank: " << m.get_rank() << ", " << target << ", "
                    << classical;
                ASSERT_NEAR(imag(state.data_cpp()[i]),
                    imag(state_ref.data_cpp()[(i + offs) % dim]), eps)
                    << "rank: " << m.get_rank() << ", " << target << ", "
                    << classical;
            }
        }
    }
}
#endif

void _ApplyTwoQubitGate(UINT n, UINT control, UINT target,
    std::function<QuantumGateBase*(UINT, UINT)>,
    std::function<Eigen::MatrixXcd(UINT, UINT, UINT)>);

TEST(GateTest_multicpu, ApplyCNOTGate_10qubit_all) {
    UINT n = 10;
    for (UINT c = 0; c < n; ++c) {
        for (UINT t = 0; t < n; ++t) {
            if (c == t) continue;
            _ApplyTwoQubitGate(
                n, c, t, gate::CNOT, get_eigen_matrix_full_qubit_CNOT);
        }
    }
}

TEST(GateTest_multicpu, ApplyCNOTGate_small) {
    Random random;
    random.set_seed(2022);
    UINT n = 2;
    _ApplyTwoQubitGate(n, 0, 1, gate::CNOT, get_eigen_matrix_full_qubit_CNOT);
    _ApplyTwoQubitGate(n, 1, 0, gate::CNOT, get_eigen_matrix_full_qubit_CNOT);
}

TEST(GateTest_multicpu, ApplyCZGate_10qubit_all) {
    UINT n = 10;
    for (UINT c = 0; c < n; ++c) {
        for (UINT t = 0; t < n; ++t) {
            if (c == t) continue;
            _ApplyTwoQubitGate(
                n, c, t, gate::CZ, get_eigen_matrix_full_qubit_CZ);
        }
    }
}

TEST(GateTest_multicpu, ApplyCZGate_small) {
    Random random;
    random.set_seed(2022);
    UINT n = 2;
    _ApplyTwoQubitGate(n, 0, 1, gate::CZ, get_eigen_matrix_full_qubit_CZ);
    _ApplyTwoQubitGate(n, 1, 0, gate::CZ, get_eigen_matrix_full_qubit_CZ);
}

TEST(GateTest_multicpu, ApplySWAPGate_10qubit_all) {
    UINT n = 10;
    for (UINT c = 0; c < n; ++c) {
        for (UINT t = 0; t < n; ++t) {
            if (c == t) continue;
            _ApplyTwoQubitGate(
                n, c, t, gate::SWAP, get_eigen_matrix_full_qubit_SWAP);
        }
    }
}

TEST(GateTest_multicpu, ApplySWAPGate_corner) {
    Random random;
    random.set_seed(2022);
    _ApplyTwoQubitGate(2, 0, 1, gate::SWAP, get_eigen_matrix_full_qubit_SWAP);
    _ApplyTwoQubitGate(2, 1, 0, gate::SWAP, get_eigen_matrix_full_qubit_SWAP);
    _ApplyTwoQubitGate(5, 4, 3, gate::SWAP, get_eigen_matrix_full_qubit_SWAP);
    _ApplyTwoQubitGate(5, 3, 4, gate::SWAP, get_eigen_matrix_full_qubit_SWAP);
}

void _ApplyTwoQubitGate(UINT n, UINT control, UINT target,
    std::function<QuantumGateBase*(UINT, UINT)> func,
    std::function<Eigen::MatrixXcd(UINT, UINT, UINT)> func_eig) {
    const ITYPE dim = 1ULL << n;

    QuantumState state_ref(n);
    QuantumState state(n, 1), test_state(n, 1);

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = inner_dim * m.get_rank();
    Eigen::VectorXcd test_state1 = Eigen::VectorXcd::Zero(dim);
    {
        if (target == control) target = (target + 1) % n;

        state_ref.set_Haar_random_state(2022);
        for (ITYPE i = 0; i < dim; ++i)
            test_state1[i] = state_ref.data_cpp()[i];
        for (ITYPE i = 0; i < inner_dim; ++i)
            state.data_cpp()[i] = state_ref.data_cpp()[(i + offs) % dim];
        // test_state.load(&state);
        for (ITYPE i = 0; i < inner_dim; ++i)
            test_state.data_cpp()[i] = state_ref.data_cpp()[(i + offs) % dim];

        // update state
        auto gate = func(control, target);
        gate->update_quantum_state(&state);
        gate->update_quantum_state(&state_ref);

        // update eigen state
        Eigen::MatrixXcd large_mat = func_eig(control, target, n);
        test_state1 = large_mat * test_state1;

        // update dense state
        ComplexMatrix small_mat;
        gate->set_matrix(small_mat);
        auto gate_dense = new QuantumGateMatrix(
            gate->target_qubit_list, small_mat, gate->control_qubit_list);
        gate_dense->update_quantum_state(&test_state);
        delete gate_dense;

        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state1[(i + offs) % dim]), 0,
                eps);
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(abs(state.data_cpp()[i] -
                            state_ref.data_cpp()[(i + offs) % dim]),
                0, eps);
    }
}

TEST(GateTest_multicpu, ApplyMultiControl) {
    const UINT n = 10;
    const ITYPE dim = 1ULL << n;
    Random random;
    random.set_seed(2023);

    QuantumState state(n, 1);
    QuantumState state_ref(n);

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = inner_dim * m.get_rank();

    for (UINT i = 0; i < 10; ++i) {
        state.set_Haar_random_state(2022);
        state_ref.load(&state);

        UINT target = random.int32() % n;
        auto gate_added_c = gate::to_matrix_gate(gate::X(target));
        UINT control = 0;
        UINT addlimit = n;
        UINT numcontrol = random.int32() % 5;
        UINT beforenum = target;

        for (UINT c = 0; c < numcontrol; ++c) {
            control =
                (beforenum + 1 + random.int32() % (addlimit - numcontrol + c)) %
                n;
            addlimit -= (beforenum - control);
            beforenum = control;
            if (control == target) control = (control + 1) % n;
            gate_added_c->add_control_qubit(control, random.int32() % 2);
        }

        // if (m.get_rank() == 0) std::cout << "# gate" << gate_added_c <<
        // std::endl;
        gate_added_c->update_quantum_state(&state);
        gate_added_c->update_quantum_state(&state_ref);
        delete gate_added_c;

        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(real(state.data_cpp()[i]),
                real(state_ref.data_cpp()[i + offs]), eps)
                << i;
        ASSERT_NEAR(imag(state.data_cpp()[i]),
            imag(state_ref.data_cpp()[i + offs]), eps)
            << i;
    }
}

void _ApplyFusedSWAPGate_multicpu(
    UINT n, UINT target0, UINT target1, UINT block_size) {
    const ITYPE dim = 1ULL << n;

    QuantumState state_ref(n);
    QuantumState state(n, 1);

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = inner_dim * m.get_rank();

    {
        state_ref.set_Haar_random_state(2022);
        state.load(&state_ref);

        // update "state_ref" using SWAP gate
        for (UINT i = 0; i < block_size; ++i) {
            auto swap_gate = gate::SWAP(target0 + i, target1 + i);
            swap_gate->update_quantum_state(&state_ref);
            delete swap_gate;
        }

        // update "state" using FusedSWAP gate
        auto bswap_gate = gate::FusedSWAP(target0, target1, block_size);
        bswap_gate->update_quantum_state(&state);
        delete bswap_gate;

        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(abs(state.data_cpp()[i] -
                            state_ref.data_cpp()[(i + offs) % dim]),
                0, eps)
                << "[rank:" << m.get_rank() << "] FusedSWAP(" << target0 << ","
                << target1 << "," << block_size << ") diff at " << i;
    }
}

TEST(GateTest_multicpu, ApplyFusedSWAPGate_10qubit_all) {
    UINT n = 10;
    for (UINT t0 = 0; t0 < n; ++t0) {
        for (UINT t1 = 0; t1 < n; ++t1) {
            if (t0 == t1) continue;
            UINT max_bs = std::min(
                (t0 < t1) ? (t1 - t0) : (t0 - t1), std::min(n - t0, n - t1));
            for (UINT bs = 1; bs <= max_bs; ++bs) {
                _ApplyFusedSWAPGate_multicpu(n, t0, t1, bs);
            }
        }
    }
}

TEST(GateTest_multicpu, ApplyMultiQubitGate) {
    const UINT n = 1;
    const ITYPE dim = 1ULL << n;

    Random random;
    QuantumState state_ref(n);
    QuantumState state(n, 1);
    std::vector<std::pair<std::function<QuantumGateBase*(UINT, UINT)>,
        std::function<Eigen::MatrixXcd(UINT, UINT, UINT)>>>
        funclist;

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = (state.outer_qc != 0) * inner_dim * m.get_rank();

    // gate::DenseMatrix
    // gate::Pauli
    // gate::PauliRotation

    Eigen::VectorXcd test_state1 = Eigen::VectorXcd::Zero(dim);
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        state_ref.set_Haar_random_state();
        state_ref.set_computational_basis(0);
        for (ITYPE i = 0; i < dim; ++i)
            test_state1[i] = state_ref.data_cpp()[i];
        for (ITYPE i = 0; i < inner_dim; ++i)
            state.data_cpp()[i] = state_ref.data_cpp()[i + offs];

        PauliOperator pauli(1.0);
        for (UINT i = 0; i < n; ++i) {
            pauli.add_single_Pauli(i, random.int32() % 4);
        }
        auto gate =
            gate::Pauli(pauli.get_index_list(), pauli.get_pauli_id_list());
        Eigen::MatrixXcd large_mat = get_eigen_matrix_full_qubit_pauli(
            pauli.get_index_list(), pauli.get_pauli_id_list(), n);
        test_state1 = large_mat * test_state1;

        std::vector<UINT> target_list, control_list;
        ComplexMatrix small_mat;
        gate->set_matrix(small_mat);
        auto gate_dense = new QuantumGateMatrix(
            gate->target_qubit_list, small_mat, gate->control_qubit_list);
        gate_dense->update_quantum_state(&state);
        delete gate_dense;

        // std::cout << state << std::endl << test_state1 << std::endl;
        // std::cout << small_mat << std::endl << large_mat << std::endl;
        // for (UINT i = 0; i < 4; ++i) std::cout << small_mat.data()[i] <<
        // std::endl;

        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state1[i + offs]), 0, eps);
    }

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        state_ref.set_Haar_random_state();
        for (ITYPE i = 0; i < dim; ++i)
            test_state1[i] = state_ref.data_cpp()[i];
        for (ITYPE i = 0; i < inner_dim; ++i)
            state.data_cpp()[i] = state_ref.data_cpp()[i + offs];

        PauliOperator pauli(1.0);
        for (UINT i = 0; i < n; ++i) {
            pauli.add_single_Pauli(i, random.int32() % 4);
        }
        double angle = random.uniform() * 3.14159;

        Eigen::MatrixXcd large_mat =
            cos(angle / 2) * Eigen::MatrixXcd::Identity(dim, dim) +
            1.i * sin(angle / 2) *
                get_eigen_matrix_full_qubit_pauli(
                    pauli.get_index_list(), pauli.get_pauli_id_list(), n);
        test_state1 = large_mat * test_state1;

        auto gate = gate::PauliRotation(
            pauli.get_index_list(), pauli.get_pauli_id_list(), angle);
        std::vector<UINT> target_list, control_list;
        ComplexMatrix small_mat;
        gate->set_matrix(small_mat);
        auto gate_dense = new QuantumGateMatrix(
            gate->target_qubit_list, small_mat, gate->control_qubit_list);
        gate_dense->update_quantum_state(&state);
        delete gate_dense;

        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state1[i + offs]), 0, eps);
    }
}

TEST(GateTest_multicpu, MergeTensorProduct) {
    UINT n = 5;
    ITYPE dim = 1ULL << n;

    auto x0 = gate::X(0);
    auto y1 = gate::Y(1);
    auto xy01 = gate::merge(x0, y1);

    QuantumState state_ref(n);
    QuantumState state(n, 1), test_state(n, 1);
    state.set_Haar_random_state(2023);

    state_ref.load(&state);
    test_state.load(&state);

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = (state.outer_qc != 0) * inner_dim * m.get_rank();

    xy01->update_quantum_state(&state_ref);
    xy01->update_quantum_state(&state);
    x0->update_quantum_state(&test_state);
    y1->update_quantum_state(&test_state);

    for (ITYPE i = 0; i < inner_dim; ++i) {
        ASSERT_NEAR(
            abs(state.data_cpp()[i] - state_ref.data_cpp()[i + offs]), 0, eps);
        ASSERT_NEAR(
            abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
    }

    delete x0;
    delete y1;
    delete xy01;
}

TEST(GateTest_multicpu, MergeMultiply) {
    UINT n = 5;
    ITYPE dim = 1ULL << n;
    auto x0 = gate::X(0);
    auto y0 = gate::Y(0);

    //  U_{z0} = YX = -iZ
    auto xy00 = gate::merge(x0, y0);

    QuantumState state_ref(n);
    QuantumState state(n, 1), test_state(n, 1);
    Eigen::VectorXcd test_state_eigen(dim);
    state.set_Haar_random_state(2023);

    state_ref.load(&state);
    test_state.load(&state);

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = (state.outer_qc != 0) * inner_dim * m.get_rank();

    xy00->update_quantum_state(&state_ref);
    xy00->update_quantum_state(&state);
    x0->update_quantum_state(&test_state);
    y0->update_quantum_state(&test_state);

    for (ITYPE i = 0; i < inner_dim; ++i) {
        ASSERT_NEAR(
            abs(state.data_cpp()[i] - state_ref.data_cpp()[i + offs]), 0, eps);
        ASSERT_NEAR(
            abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
    }

    delete x0;
    delete y0;
    delete xy00;
}

TEST(GateTest_multicpu, MergeTensorProductAndMultiply) {
    UINT n = 5;
    ITYPE dim = 1ULL << n;

    auto x0 = gate::X(0);
    auto y1 = gate::Y(1);
    auto xy01 = gate::merge(x0, y1);
    // std::cout << xy01 << std::endl;
    auto iy01 = gate::merge(xy01, x0);

    // Expected : x_0 y_1 x_0 = y_1

    QuantumState state_ref(n);
    QuantumState state(n, 1), test_state(n, 1);
    state.set_Haar_random_state();

    state_ref.load(&state);
    test_state.load(&state);

    MPIutil& m = MPIutil::get_inst();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = (state.outer_qc != 0) * inner_dim * m.get_rank();

    iy01->update_quantum_state(&state);
    iy01->update_quantum_state(&state_ref);
    y1->update_quantum_state(&test_state);

    for (ITYPE i = 0; i < inner_dim; ++i) {
        ASSERT_NEAR(
            abs(state.data_cpp()[i] - state_ref.data_cpp()[i + offs]), 0, eps);
        ASSERT_NEAR(
            abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
    }

    delete x0;
    delete y1;
    delete xy01;
    delete iy01;
}

TEST(GateTest_multicpu, RandomPauliMerge) {
    MPIutil& m = MPIutil::get_inst();
    const UINT num_global_qubit = (UINT)std::log2(m.get_size());
    UINT n = 10 + num_global_qubit;
    ITYPE dim = 1ULL << n;

    UINT gate_count = 10;
    UINT max_repeat = 3;
    Random random;
    random.set_seed(2);

    std::vector<UINT> new_pauli_ids = {0, 0, 0, 1};
    std::vector<UINT> targets = {0, 1, 2, 2};

    // define states
    QuantumState state_ref(n);
    QuantumState state(n, 1), test_state(n, 1);

    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = (state.outer_qc != 0) * inner_dim * m.get_rank();

    for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
        // pick random state and copy to test
        // state_ref.set_Haar_random_state(2022 + repeat);
        state.set_Haar_random_state();
        test_state.load(&state);
        state_ref.load(&state);

        // check equivalence
        for (ITYPE i = 0; i < inner_dim; ++i) {
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - state_ref.data_cpp()[i + offs]), 0,
                eps);
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        }

        QuantumGateBase* merged_gate = gate::Identity(0);
        QuantumGateMatrix* next_merged_gate = NULL;
        QuantumGateBase* new_gate = NULL;
        // std::cout << "initial state : " << state << std::endl;

        for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
            // pick random pauli
            UINT new_pauli_id = random.int32() % 4;
            UINT target = random.int32() % n;
            // std::cout << "#add_gate " << m.get_rank() << ", " << repeat <<
            // ", "
            //           << gate_index << ", " << new_pauli_id << ", " << target
            //           << std::endl;
            // UINT new_pauli_id = new_pauli_ids[gate_index];
            // UINT target = targets[gate_index];
            if (new_pauli_id == 0)
                new_gate = gate::Identity(target);
            else if (new_pauli_id == 1)
                new_gate = gate::X(target);
            else if (new_pauli_id == 2)
                new_gate = gate::Y(target);
            else if (new_pauli_id == 3)
                new_gate = gate::Z(target);
            else
                FAIL();

            // create new gate with merge
            next_merged_gate = gate::merge(merged_gate, new_gate);
            delete merged_gate;
            merged_gate = next_merged_gate;
            next_merged_gate = NULL;

            // update test state with latest gate
            new_gate->update_quantum_state(&test_state);

            // dispose picked pauli
            delete new_gate;
        }
        merged_gate->update_quantum_state(&state);
        merged_gate->update_quantum_state(&state_ref);
        delete merged_gate;

        // check equivalence
        for (ITYPE i = 0; i < inner_dim; ++i) {
            ASSERT_NEAR(
                abs(state_ref.data_cpp()[i + offs] - state.data_cpp()[i]), 0,
                eps);
            ASSERT_NEAR(
                abs(state_ref.data_cpp()[i + offs] - test_state.data_cpp()[i]),
                0, eps);
        }
    }
}

TEST(GateTest_multicpu, RandomPauliRotationMerge) {
    MPIutil& m = MPIutil::get_inst();
    const UINT num_global_qubit = (UINT)std::log2(m.get_size());
    UINT n = 6 + num_global_qubit;
    ITYPE dim = 1ULL << n;

    UINT gate_count = 5;
    UINT max_repeat = 2;
    Random random;
    random.set_seed(2);

    std::vector<UINT> new_pauli_ids = {0, 0, 0, 1};
    std::vector<UINT> targets = {0, 1, 2, 2};

    // define states
    QuantumState state_ref(n);
    QuantumState state(n, 1), test_state(n, 1);
    Eigen::VectorXcd test_state_eigen(dim);

    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = (state.outer_qc != 0) * inner_dim * m.get_rank();

    for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
        // pick random state and copy to test
        state_ref.set_Haar_random_state(2022);
        test_state.load(&state_ref);
        for (ITYPE i = 0; i < dim; ++i)
            test_state_eigen[i] = state_ref.data_cpp()[i];
        for (ITYPE i = 0; i < inner_dim; ++i)
            state.data_cpp()[i] = state_ref.data_cpp()[i + offs];

        // check equivalence
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state_eigen[i + offs]), 0, eps);
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(test_state.data_cpp()[i] - test_state_eigen[i + offs]), 0,
                eps);

        QuantumGateBase* merged_gate = gate::Identity(0);
        QuantumGateMatrix* next_merged_gate = NULL;
        QuantumGateBase* new_gate = NULL;
        Eigen::MatrixXcd total_matrix = Eigen::MatrixXcd::Identity(dim, dim);
        // std::cout << "initial state : " << state << std::endl;

        for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
            // pick random pauli
            UINT new_pauli_id = (random.int32() % 3) + 1;
            UINT target = random.int32() % n;
            double angle = random.uniform() * 3.14159;
            // UINT new_pauli_id = new_pauli_ids[gate_index];
            // UINT target = targets[gate_index];
            if (new_pauli_id == 1)
                new_gate = gate::RX(target, angle);
            else if (new_pauli_id == 2)
                new_gate = gate::RY(target, angle);
            else if (new_pauli_id == 3)
                new_gate = gate::RZ(target, angle);
            else
                FAIL();

            // create new gate with merge
            next_merged_gate = gate::merge(merged_gate, new_gate);
            delete merged_gate;
            merged_gate = next_merged_gate;
            next_merged_gate = NULL;

            // update test state with latest gate
            new_gate->update_quantum_state(&test_state);

            // update eigen state with matrix mul
            auto new_gate_matrix =
                get_expanded_eigen_matrix_with_identity(target,
                    cos(angle / 2) * ComplexMatrix::Identity(2, 2) +
                        1.i * sin(angle / 2) *
                            get_eigen_matrix_single_Pauli(new_pauli_id),
                    n);
            total_matrix = new_gate_matrix * total_matrix;
            test_state_eigen = new_gate_matrix * test_state_eigen;

            ComplexMatrix check_mat;
            merged_gate->set_matrix(check_mat);
            if (check_mat.rows() == total_matrix.rows()) {
                for (ITYPE x = 0; x < dim; ++x) {
                    for (ITYPE y = 0; y < dim; ++y) {
                        ASSERT_NEAR(
                            abs(total_matrix(x, y) - check_mat(x, y)), 0, eps)
                            << (QuantumGateMatrix*)merged_gate << std::endl
                            << "current eigen matrix : \n"
                            << total_matrix << std::endl;
                    }
                }
            }

            // dispose picked pauli
            delete new_gate;
        }
        merged_gate->update_quantum_state(&state);
        delete merged_gate;
        // check equivalence
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(test_state.data_cpp()[i] - test_state_eigen[i + offs]), 0,
                eps);
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state_eigen[i + offs]), 0, eps);
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
    }
}

TEST(GateTest_multicpu, RandomUnitaryMerge) {
    MPIutil& m = MPIutil::get_inst();
    const UINT num_global_qubit = (UINT)std::log2(m.get_size());
    UINT n = 5 + num_global_qubit;
    ITYPE dim = 1ULL << n;

    UINT gate_count = 5;
    UINT max_repeat = 2;
    Random random;
    random.set_seed(2);

    std::vector<UINT> new_pauli_ids = {0, 0, 0, 1};
    std::vector<UINT> targets = {0, 1, 2, 2};

    // define states
    QuantumState state_ref(n);
    QuantumState state(n, 1), test_state(n, 1);
    Eigen::VectorXcd test_state_eigen(dim);

    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = (state.outer_qc != 0) * inner_dim * m.get_rank();

    for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
        // pick random state and copy to test
        state_ref.set_Haar_random_state(2022);
        test_state.load(&state_ref);
        for (ITYPE i = 0; i < dim; ++i)
            test_state_eigen[i] = state_ref.data_cpp()[i];
        for (ITYPE i = 0; i < inner_dim; ++i)
            state.data_cpp()[i] = state_ref.data_cpp()[i + offs];

        // check equivalence
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state_eigen[i + offs]), 0, eps);
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(test_state.data_cpp()[i] - test_state_eigen[i + offs]), 0,
                eps);

        QuantumGateBase* merged_gate = gate::Identity(0);
        QuantumGateMatrix* next_merged_gate = NULL;
        QuantumGateBase* new_gate = NULL;
        Eigen::MatrixXcd total_matrix = Eigen::MatrixXcd::Identity(dim, dim);
        // std::cout << "initial state : " << state << std::endl;

        for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
            // pick random pauli
            UINT new_pauli_id = (random.int32() % 3) + 1;
            UINT target = random.int32() % n;
            double di = random.uniform();
            double dx = random.uniform();
            double dy = random.uniform();
            double dz = random.uniform();
            double norm = sqrt(di * di + dx * dx + dy * dy + dz * dz);
            di /= norm;
            dx /= norm;
            dy /= norm;
            dz /= norm;
            ComplexMatrix mat =
                di * get_eigen_matrix_single_Pauli(0) +
                1.i * (dx * get_eigen_matrix_single_Pauli(1) +
                          dy * get_eigen_matrix_single_Pauli(2) +
                          dz * get_eigen_matrix_single_Pauli(3));

            auto new_gate = gate::DenseMatrix(target, mat);

            // create new gate with merge
            next_merged_gate = gate::merge(merged_gate, new_gate);
            delete merged_gate;
            merged_gate = next_merged_gate;
            next_merged_gate = NULL;

            // update test state with latest gate
            new_gate->update_quantum_state(&test_state);

            // update eigen state with matrix mul
            auto new_gate_matrix =
                get_expanded_eigen_matrix_with_identity(target, mat, n);
            total_matrix = new_gate_matrix * total_matrix;
            test_state_eigen = new_gate_matrix * test_state_eigen;

            ComplexMatrix check_mat;
            merged_gate->set_matrix(check_mat);
            if (check_mat.rows() == total_matrix.rows()) {
                for (ITYPE x = 0; x < dim; ++x) {
                    for (ITYPE y = 0; y < dim; ++y) {
                        ASSERT_NEAR(
                            abs(total_matrix(x, y) - check_mat(x, y)), 0, eps)
                            << (QuantumGateMatrix*)merged_gate << std::endl
                            << "current eigen matrix : \n"
                            << total_matrix << std::endl;
                    }
                }
            }

            // dispose picked pauli
            delete new_gate;
        }
        merged_gate->update_quantum_state(&state);
        delete merged_gate;
        // check equivalence
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(test_state.data_cpp()[i] - test_state_eigen[i + offs]), 0,
                eps);
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state_eigen[i + offs]), 0, eps);
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
    }
}

TEST(GateTest_multicpu, RandomUnitaryMergeLarge) {
    MPIutil& m = MPIutil::get_inst();
    const UINT num_global_qubit = (UINT)std::log2(m.get_size());
    UINT n = 10 + num_global_qubit;
    ITYPE dim = 1ULL << n;

    UINT gate_count = 5;
    UINT max_repeat = 2;
    Random random;
    random.set_seed(2);

    std::vector<UINT> new_pauli_ids = {0, 0, 0, 1};
    std::vector<UINT> targets = {0, 1, 2, 2};

    // define states
    QuantumState state_ref(n);
    QuantumState state(n, 1), test_state(n, 1);
    Eigen::VectorXcd test_state_eigen(dim);

    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = (state.outer_qc != 0) * inner_dim * m.get_rank();

    for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
        // pick random state and copy to test
        state_ref.set_Haar_random_state(2022);
        test_state.load(&state_ref);
        for (ITYPE i = 0; i < dim; ++i)
            test_state_eigen[i] = state_ref.data_cpp()[i];
        for (ITYPE i = 0; i < inner_dim; ++i)
            state.data_cpp()[i] = state_ref.data_cpp()[i + offs];

        // check equivalence
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state_eigen[i + offs]), 0, eps);
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(test_state.data_cpp()[i] - test_state_eigen[i + offs]), 0,
                eps);

        QuantumGateBase* merged_gate1 = gate::Identity(0);
        QuantumGateBase* merged_gate2 = gate::Identity(0);
        QuantumGateMatrix* next_merged_gate = NULL;
        QuantumGateBase* new_gate = NULL;
        for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
            // pick random pauli
            UINT new_pauli_id = (random.int32() % 3) + 1;
            UINT target = random.int32() % n;
            double di = random.uniform();
            double dx = random.uniform();
            double dy = random.uniform();
            double dz = random.uniform();
            double norm = sqrt(di * di + dx * dx + dy * dy + dz * dz);
            di /= norm;
            dx /= norm;
            dy /= norm;
            dz /= norm;
            ComplexMatrix mat =
                di * get_eigen_matrix_single_Pauli(0) +
                1.i * (dx * get_eigen_matrix_single_Pauli(1) +
                          dy * get_eigen_matrix_single_Pauli(2) +
                          dz * get_eigen_matrix_single_Pauli(3));

            auto new_gate = gate::DenseMatrix(target, mat);

            // create new gate with merge
            next_merged_gate = gate::merge(merged_gate1, new_gate);
            delete merged_gate1;
            merged_gate1 = next_merged_gate;
            next_merged_gate = NULL;

            // dispose picked pauli
            delete new_gate;
        }
        for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
            // pick random pauli
            UINT new_pauli_id = (random.int32() % 3) + 1;
            UINT target = random.int32() % n;
            double di = random.uniform();
            double dx = random.uniform();
            double dy = random.uniform();
            double dz = random.uniform();
            double norm = sqrt(di * di + dx * dx + dy * dy + dz * dz);
            di /= norm;
            dx /= norm;
            dy /= norm;
            dz /= norm;
            ComplexMatrix mat =
                di * get_eigen_matrix_single_Pauli(0) +
                1.i * (dx * get_eigen_matrix_single_Pauli(1) +
                          dy * get_eigen_matrix_single_Pauli(2) +
                          dz * get_eigen_matrix_single_Pauli(3));

            auto new_gate = gate::DenseMatrix(target, mat);

            // create new gate with merge
            next_merged_gate = gate::merge(merged_gate2, new_gate);
            delete merged_gate2;
            merged_gate2 = next_merged_gate;
            next_merged_gate = NULL;

            // dispose picked pauli
            delete new_gate;
        }
        QuantumGateBase* merged_gate = gate::merge(merged_gate1, merged_gate2);
        merged_gate->update_quantum_state(&state);
        merged_gate1->update_quantum_state(&test_state);
        merged_gate2->update_quantum_state(&test_state);

        delete merged_gate;
        delete merged_gate1;
        delete merged_gate2;
        // check equivalence
        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
    }
}

TEST(GateTest_multicpu, RandomControlMerge) {
    MPIutil& m = MPIutil::get_inst();
    const UINT num_global_qubit = (UINT)std::log2(m.get_size());
    UINT n = 10 + num_global_qubit;
    ITYPE dim = 1ULL << n;

    UINT gate_count = 10;
    Random random;
    random.set_seed(2023);
    std::mt19937 eng(2020);

    QuantumState dum_state(n, 1);
    const ITYPE inner_dim = dim >> dum_state.outer_qc;
    const ITYPE offs = (dum_state.outer_qc != 0) * inner_dim * m.get_rank();

    for (gate_count = 1; gate_count < n * 2; ++gate_count) {
        std::vector<UINT> arr;
        // for (UINT i = 0; i < 5; ++i) arr.push_back(i); // max 5 qubit
        // dense-matrix-gate
        for (UINT i = 0; i < n; ++i)
            arr.push_back(i);  // max 5 qubit dense-matrix-gate
        std::shuffle(arr.begin(), arr.end(), eng);
        arr.resize(5);

        QuantumState ref_state(n);
        QuantumState state(n, 1), test_state(n, 1);
        state.set_Haar_random_state();
        ref_state.load(&state);
        test_state.load(&state);
        QuantumGateBase* merge_gate1 = gate::Identity(0);
        QuantumGateBase* merge_gate2 = gate::Identity(0);

        for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
            std::shuffle(arr.begin(), arr.end(), eng);
            UINT target = arr[0];
            UINT control = arr[1];
            auto new_gate = gate::CNOT(control, target);
            merge_gate1 = gate::merge(merge_gate1, new_gate);

            new_gate->update_quantum_state(&test_state);
        }
        merge_gate1->update_quantum_state(&state);
        merge_gate1->update_quantum_state(&ref_state);

        for (ITYPE i = 0; i < inner_dim; ++i) {
            ASSERT_NEAR(
                abs(ref_state.data_cpp()[i + offs] - test_state.data_cpp()[i]),
                0, eps)
                << m.get_rank() << ", " << i << ", " << offs;
            //    << ref_state << "\n\n"
            //    << test_state << "\n";
            ASSERT_NEAR(
                abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps);
            //    << state << "\n\n"
            //    << test_state << "\n";
        }
    }
}

TEST(GateTest_multicpu, RandomUnitaryMergeWithControl) {
    MPIutil& m = MPIutil::get_inst();
    const ITYPE mpisize = m.get_size();
    const UINT num_global_qubit = (UINT)std::log2(m.get_size());
    UINT n = 10 + num_global_qubit;
    ITYPE dim = 1ULL << n;

    Random random;
    random.set_seed(2023);
    std::mt19937 eng(2020);

    QuantumState dum_state(n, 1);
    const ITYPE inner_dim = dim >> dum_state.outer_qc;
    const ITYPE offs = (dum_state.outer_qc != 0) * inner_dim * m.get_rank();

    for (UINT gate_count = 1; gate_count < n * 2; ++gate_count) {
        // max 5 qubit dense-matrix-gate
        std::vector<UINT> arr;
        for (UINT i = 0; i < n; ++i) arr.push_back(i);
        std::shuffle(arr.begin(), arr.end(), eng);
        UINT control = arr[5];
        arr.resize(5);

        QuantumState ref_state(n);
        QuantumState state(n, 1);
        state.set_Haar_random_state();
        ref_state.load(&state);
        QuantumGateBase* merge_gate1 = gate::H(arr[0]);

        for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
            std::shuffle(arr.begin(), arr.end(), eng);
            UINT target1 = arr[0];
            UINT target2 = arr[1];
            UINT seed = 2020 + gate_count * 100 + gate_index;
            auto new_gate = gate::RandomUnitary({target1, target2}, seed);
            merge_gate1 = gate::merge(merge_gate1, new_gate);
        }
        auto merge_gate = gate::to_matrix_gate(merge_gate1);
        merge_gate->add_control_qubit(control, random.int32() % 2);
        merge_gate->update_quantum_state(&state);
        merge_gate->update_quantum_state(&ref_state);

        for (ITYPE i = 0; i < inner_dim; ++i) {
            // ASSERT_NEAR(abs(ref_state.data_cpp()[i + offs] -
            // state.data_cpp()[i]), 0, eps)
            ASSERT_NEAR(real(ref_state.data_cpp()[i + offs]),
                real(state.data_cpp()[i]), eps)
                << m.get_rank() << ", " << i << ", " << offs;
            ASSERT_NEAR(imag(ref_state.data_cpp()[i + offs]),
                imag(state.data_cpp()[i]), eps)
                << m.get_rank() << ", " << i << ", " << offs;
        }
    }
}

TEST(GateTest_multicpu, RandomUnitaryMergeWithTwoControl) {
    MPIutil& m = MPIutil::get_inst();
    const UINT num_global_qubit = (UINT)std::log2(m.get_size());
    UINT n = 10 + num_global_qubit;
    ITYPE dim = 1ULL << n;

    Random random;
    random.set_seed(2024);
    std::mt19937 eng(2025);

    QuantumState dum_state(n, 1);
    const ITYPE inner_dim = dim >> dum_state.outer_qc;
    const ITYPE offs = (dum_state.outer_qc != 0) * inner_dim * m.get_rank();

    for (UINT gate_count = 1; gate_count < n * 2; ++gate_count) {
        // max 5 qubit dense-matrix-gate
        std::vector<UINT> arr;
        for (UINT i = 0; i < n; ++i) arr.push_back(i);
        std::shuffle(arr.begin(), arr.end(), eng);
        UINT control1 = arr[5];
        UINT control2 = arr[6];
        arr.resize(5);

        QuantumState ref_state(n);
        QuantumState state(n, 1);
        state.set_Haar_random_state();
        ref_state.load(&state);
        QuantumGateBase* merge_gate1 = gate::H(arr[0]);

        for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
            std::shuffle(arr.begin(), arr.end(), eng);
            UINT target1 = arr[0];
            UINT target2 = arr[1];
            UINT seed = 2030 + gate_count * 100 + gate_index;
            auto new_gate = gate::RandomUnitary({target1, target2}, seed);
            merge_gate1 = gate::merge(merge_gate1, new_gate);
        }
        auto merge_gate = gate::to_matrix_gate(merge_gate1);
        merge_gate->add_control_qubit(control1, random.int32() % 2);
        merge_gate->add_control_qubit(control2, random.int32() % 2);
        merge_gate->update_quantum_state(&state);
        merge_gate->update_quantum_state(&ref_state);

        for (ITYPE i = 0; i < inner_dim; ++i) {
            // ASSERT_NEAR(abs(ref_state.data_cpp()[i + offs] -
            // state.data_cpp()[i]), 0, eps)
            ASSERT_NEAR(real(ref_state.data_cpp()[i + offs]),
                real(state.data_cpp()[i]), eps)
                << m.get_rank() << ", " << i << ", " << offs;
            ASSERT_NEAR(imag(ref_state.data_cpp()[i + offs]),
                imag(state.data_cpp()[i]), eps)
                << m.get_rank() << ", " << i << ", " << offs;
        }
    }
}

TEST(GateTest_multicpu, ProbabilisticGate) {
    auto gate1 = gate::X(0);
    auto gate2 = gate::X(1);
    auto gate3 = gate::X(2);
    auto prob_gate =
        gate::Probabilistic({0.25, 0.25, 0.25}, {gate1, gate2, gate2});
    QuantumState s(3, 1);
    s.set_computational_basis(0);
    prob_gate->update_quantum_state(&s);
    delete gate1;
    delete gate2;
    delete gate3;
    delete prob_gate;
}

/*
TEST(GateTest_multicpu, CPTPGate) {
    auto gate1 = gate::merge(gate::P0(0), gate::P0(1));
    auto gate2 = gate::merge(gate::P0(0), gate::P1(1));
    auto gate3 = gate::merge(gate::P1(0), gate::P0(1));
    auto gate4 = gate::merge(gate::P1(0), gate::P1(1));

    auto CPTP = gate::CPTP({ gate3, gate2, gate1, gate4 });
    QuantumState s(3, 1);
    s.set_computational_basis(0);
    CPTP->update_quantum_state(&s);
    s.set_Haar_random_state();
    CPTP->update_quantum_state(&s);
    delete CPTP;
}

TEST(GateTest_multicpu, InstrumentGate) {
    auto gate1 = gate::merge(gate::P0(0), gate::P0(1));
    auto gate2 = gate::merge(gate::P0(0), gate::P1(1));
    auto gate3 = gate::merge(gate::P1(0), gate::P0(1));
    auto gate4 = gate::merge(gate::P1(0), gate::P1(1));

    auto Inst = gate::Instrument({ gate3, gate2, gate1, gate4 }, 1);
    QuantumState s(3, 1);
    s.set_computational_basis(0);
    Inst->update_quantum_state(&s);
    UINT res1 = s.get_classical_value(1);
    ASSERT_EQ(res1, 2);
    s.set_Haar_random_state();
    Inst->update_quantum_state(&s);
    UINT res2 = s.get_classical_value(1);
    delete Inst;
}

TEST(GateTest_multicpu, AdaptiveGate) {
    auto x = gate::X(0);
    auto adaptive = gate::Adaptive(
        x, [](const std::vector<UINT>& vec) { return vec[2] == 1; });
    QuantumState s(1, 1);
    s.set_computational_basis(0);
    s.set_classical_value(2, 1);
    adaptive->update_quantum_state(&s);
    s.set_classical_value(2, 0);
    adaptive->update_quantum_state(&s);
    delete adaptive;
}

// When using multi-cpu, unitary gate operation for all qubits is not possible.
TEST(GateTest_multicpu, RandomUnitaryGate) {
    for (UINT qubit_count = 1; qubit_count < 5; ++qubit_count) {
        ITYPE dim = 1ULL << qubit_count;
        std::vector<UINT> target_qubit_list;
        for (UINT i = 0; i < qubit_count; ++i) {
            target_qubit_list.push_back(i);
        }
        auto gate = gate::RandomUnitary(target_qubit_list);
        ComplexMatrix cm;
        gate->set_matrix(cm);
        auto eye = cm * cm.adjoint();
        for (ITYPE i = 0; i < dim; ++i) {
            for (ITYPE j = 0; j < dim; ++j) {
                if (i == j) {
                    ASSERT_NEAR(abs(eye(i, j)), 1, eps);
                } else {
                    ASSERT_NEAR(abs(eye(i, j)), 0, eps);
                }
            }
        }
    }
}

TEST(GateTest_multicpu, ReversibleBooleanGate) {
        std::function<ITYPE(ITYPE,ITYPE)> func = [](ITYPE index, ITYPE dim) ->
ITYPE { return (index + 1) % dim;
        };
        std::vector<UINT> target_qubit = { 2,0 };
        auto gate = gate::ReversibleBoolean(target_qubit, func);
        ComplexMatrix cm;
        gate->set_matrix(cm);
        QuantumState state(3, 1);
        gate->update_quantum_state(&state);
        ASSERT_NEAR(abs(state.data_cpp()[4]-1.), 0, eps);
        gate->update_quantum_state(&state);
        ASSERT_NEAR(abs(state.data_cpp()[1] - 1.), 0, eps);
        gate->update_quantum_state(&state);
        ASSERT_NEAR(abs(state.data_cpp()[5] - 1.), 0, eps);
        gate->update_quantum_state(&state);
        ASSERT_NEAR(abs( state.data_cpp()[0]-1.),0, eps);
*/
/*
std::cout << state.to_string() << std::endl;
gate->update_quantum_state(&state);
std::cout << state.to_string() << std::endl;
gate->update_quantum_state(&state);
std::cout << state.to_string() << std::endl;
gate->update_quantum_state(&state);
std::cout << state.to_string() << std::endl;
gate->update_quantum_state(&state);
std::cout << state.to_string() << std::endl;
gate->update_quantum_state(&state);
std::cout << state.to_string() << std::endl;
gate->update_quantum_state(&state);
std::cout << state.to_string() << std::endl;
gate->update_quantum_state(&state);
std::cout << state.to_string() << std::endl;
gate->update_quantum_state(&state);
std::cout << state.to_string() << std::endl;
gate->update_quantum_state(&state);
std::cout << state.to_string() << std::endl;
gate->update_quantum_state(&state);
std::cout << state.to_string() << std::endl;
gate->update_quantum_state(&state);
std::cout << state.to_string() << std::endl;
*/
/*
}

TEST(GateTest_multicpu, TestNoise) {
        const UINT n = 10;
        QuantumState state(n, 1);
        Random random;
        auto bitflip = gate::BitFlipNoise(0, random.uniform());
        auto dephase = gate::DephasingNoise(0, random.uniform());
        auto independetxz = gate::IndependentXZNoise(0, random.uniform());
        auto depolarizing = gate::DepolarizingNoise(0, random.uniform());
        auto amp_damp = gate::AmplitudeDampingNoise(0, random.uniform());
        auto measurement = gate::Measurement(0, 0);
        bitflip->update_quantum_state(&state);
        dephase->update_quantum_state(&state);
        independetxz->update_quantum_state(&state);
        depolarizing->update_quantum_state(&state);
        amp_damp->update_quantum_state(&state);
        measurement->update_quantum_state(&state);
        delete bitflip;
        delete dephase;
        delete independetxz;
        delete depolarizing;
        delete amp_damp;
        delete measurement;
}

TEST(GateTest_multicpu, DuplicateIndex) {
        {
                auto gate1 = gate::CNOT(10,13);
                EXPECT_TRUE(gate1 != NULL);
                delete gate1;
                auto gate2 = gate::CNOT(21,21);
                ASSERT_EQ(NULL, gate2);
        }
        {
                auto gate1 = gate::CZ(10, 13);
                EXPECT_TRUE(gate1 != NULL);
                delete gate1;
                auto gate2 = gate::CZ(21, 21);
                ASSERT_EQ(NULL, gate2);
        }
        {
                auto gate1 = gate::SWAP(10, 13);
                EXPECT_TRUE(gate1 != NULL);
                delete gate1;
                auto gate2 = gate::SWAP(21, 21);
                ASSERT_EQ(NULL, gate2);
        }
        {
                auto gate1 = gate::Pauli({ 2,1,0,3,7,9,4 }, { 0,0,0,0,0,0,0 });
                EXPECT_TRUE(gate1 != NULL);
                delete gate1;
                auto gate2 = gate::Pauli({ 0,1,3,1,5,6,2 }, { 0,0,0,0,0,0,0 });
                ASSERT_EQ(NULL, gate2);
        }
        {
                auto gate1 = gate::PauliRotation({ 2,1,0,3,7,9,4 }, {
0,0,0,0,0,0,0 }, 0.0); EXPECT_TRUE(gate1 != NULL); delete gate1; auto gate2 =
gate::PauliRotation({ 0,1,3,1,5,6,2 }, { 0,0,0,0,0,0,0 }, 0.0); ASSERT_EQ(NULL,
gate2);
        }
        {
                auto gate1 = gate::DenseMatrix({ 10, 13 },
ComplexMatrix::Identity(4,4)); EXPECT_TRUE(gate1 != NULL); delete gate1; auto
gate2 = gate::DenseMatrix({ 21, 21 }, ComplexMatrix::Identity(4, 4));
                ASSERT_EQ(NULL, gate2);
        }
        {
                auto matrix = SparseComplexMatrix(4, 4);
                matrix.setIdentity();
                auto gate1 = gate::SparseMatrix({ 10, 13 }, matrix);
                EXPECT_TRUE(gate1 != NULL);
                delete gate1;
                auto gate2 = gate::SparseMatrix({ 21, 21 }, matrix);
                ASSERT_EQ(NULL, gate2);
        }
        {
                auto gate1 = gate::RandomUnitary({ 10, 13 });
                EXPECT_TRUE(gate1 != NULL);
                delete gate1;
                auto gate2 = gate::RandomUnitary({ 21, 21 });
                ASSERT_EQ(NULL, gate2);
        }
        {
                auto ident = [](ITYPE a, ITYPE dim) {return a; };
                auto gate1 = gate::ReversibleBoolean({ 10, 13 },ident);
                EXPECT_TRUE(gate1 != NULL);
                delete gate1;
                auto gate2 = gate::ReversibleBoolean({ 21, 21 },ident);
                ASSERT_EQ(NULL, gate2);
        }
        {
                auto gate1 = gate::TwoQubitDepolarizingNoise(10,13,0.1);
                EXPECT_TRUE(gate1 != NULL);
                delete gate1;
                auto gate2 = gate::TwoQubitDepolarizingNoise(21,21,0.1);
                ASSERT_EQ(NULL, gate2);
        }
}
*/
#endif  // #ifdef _USE_MPI
