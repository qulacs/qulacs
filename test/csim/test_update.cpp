
#include <gtest/gtest.h>
#include "../util/util.h"
#include <Eigen/Core>
#include <string>
#include <algorithm>

#ifndef _MSC_VER
extern "C" {
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
}
#else
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
#endif



TEST(UpdateTest, SingleQubitGateTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    Eigen::MatrixXcd X(2, 2), Y(2, 2), Z(2, 2), H(2, 2), S(2, 2), T(2, 2), sqrtX(2, 2), sqrtY(2, 2);
    X << 0, 1, 1, 0;
    Y << 0, -1.i, 1.i, 0;
    Z << 1, 0, 0, -1;
    H << 1, 1, 1, -1; H /= sqrt(2.);
    S << 1, 0, 0, 1.i;
    T << 1, 0, 0, (1. + 1.i) / sqrt(2.);
    sqrtX << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
    sqrtY << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;

    UINT target;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];
    typedef std::tuple<std::function<void(UINT, CTYPE*, ITYPE)>, Eigen::MatrixXcd, std::string> testset;
    std::vector<testset> test_list;
    test_list.push_back(std::make_tuple(X_gate, X, "X"));
    test_list.push_back(std::make_tuple(Y_gate, Y, "Y"));
    test_list.push_back(std::make_tuple(Z_gate, Z, "Z"));
    test_list.push_back(std::make_tuple(S_gate, S, "S"));
    test_list.push_back(std::make_tuple(Sdag_gate, S.adjoint(), "Sdag"));
    test_list.push_back(std::make_tuple(T_gate, T, "T"));
    test_list.push_back(std::make_tuple(Tdag_gate, T.adjoint(), "Tdag"));
    test_list.push_back(std::make_tuple(sqrtX_gate, sqrtX, "sqrtX"));
    test_list.push_back(std::make_tuple(sqrtXdag_gate, sqrtX.adjoint(), "sqrtXdag"));
    test_list.push_back(std::make_tuple(sqrtY_gate, sqrtY, "sqrtY"));
    test_list.push_back(std::make_tuple(sqrtYdag_gate, sqrtY.adjoint(), "sqrtYdag"));

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        for (auto tup : test_list) {
            target = rand_int(n);
            auto func = std::get<0>(tup);
            auto mat = std::get<1>(tup);
            auto name = std::get<2>(tup);
            func(target, state, dim);
            test_state = get_expanded_eigen_matrix_with_identity(target, mat, n) * test_state;
            state_equal(state, test_state, dim, name);
        }
    }
    release_quantum_state(state);
}


TEST(UpdateTest, ProjectionAndNormalizeTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;
    const double eps = 1e-14;

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    UINT target;
    double prob;

    auto state = allocate_quantum_state(dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state, dim);
        Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
        for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

        // Z-projection operators 
        target = rand_int(n);
        if (rep % 2 == 0) {
            prob = M0_prob(target, state, dim);
            EXPECT_GT(prob, 1e-10);
            P0_gate(target, state, dim);
            ASSERT_NEAR(state_norm(state, dim), prob, eps);
            normalize(prob, state, dim);

            test_state = get_expanded_eigen_matrix_with_identity(target, P0, n)*test_state;
            ASSERT_NEAR(test_state.squaredNorm(), prob, eps);
            test_state.normalize();
            state_equal(state, test_state, dim, "P0 gate");
        }
        else {
            prob = M1_prob(target, state, dim);
            EXPECT_GT(prob, 1e-10);
            P1_gate(target, state, dim);
            ASSERT_NEAR(state_norm(state, dim), prob, eps);
            normalize(prob, state, dim);

            test_state = get_expanded_eigen_matrix_with_identity(target, P1, n)*test_state;
            ASSERT_NEAR(test_state.squaredNorm(), prob, eps);
            test_state.normalize();
            state_equal(state, test_state, dim, "P1 gate");
        }
    }
    release_quantum_state(state);
}

TEST(UpdateTest, SingleQubitRotationGateTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    Eigen::MatrixXcd Identity(2,2),X(2, 2), Y(2, 2), Z(2, 2);
    Identity << 1, 0, 0, 1;
    X << 0, 1, 1, 0;
    Y << 0, -1.i, 1.i, 0;
    Z << 1, 0, 0, -1;

    UINT target;
    double angle;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];
    typedef std::tuple<std::function<void(UINT, double, CTYPE*, ITYPE)>, Eigen::MatrixXcd, std::string> testset;
    std::vector<testset> test_list;
    test_list.push_back(std::make_tuple(RX_gate, X, "Xrot"));
    test_list.push_back(std::make_tuple(RY_gate, Y, "Yrot"));
    test_list.push_back(std::make_tuple(RZ_gate, Z, "Zrot"));

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        for (auto tup : test_list) {
            target = rand_int(n);
            angle = rand_real();
            auto func = std::get<0>(tup);
            auto mat = std::get<1>(tup);
            auto name = std::get<2>(tup);
            func(target, angle, state, dim);
            test_state = get_expanded_eigen_matrix_with_identity(target, cos(angle/2)*Identity + 1.i*sin(angle/2)*mat, n) * test_state;
            state_equal(state, test_state, dim, name);
        }
    }
    release_quantum_state(state);
}





TEST(UpdateTest, TwoQubitGateTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    UINT target,control;
    
    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];
    typedef std::tuple<
        std::function<void(UINT, UINT, CTYPE*, ITYPE)>, 
        std::function<Eigen::MatrixXcd(UINT,UINT,UINT)>, 
        std::string> testset;
    std::vector<testset> test_list;
    test_list.push_back(std::make_tuple(CNOT_gate, get_eigen_matrix_full_qubit_CNOT, "CNOT"));
    test_list.push_back(std::make_tuple(CZ_gate, get_eigen_matrix_full_qubit_CZ, "CZ"));
    test_list.push_back(std::make_tuple(SWAP_gate, get_eigen_matrix_full_qubit_SWAP, "SWAP"));

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        for (auto tup : test_list) {
            std::random_shuffle(index_list.begin(), index_list.end());
            target = index_list[0];
            control = index_list[1];
            ASSERT_NE(target, control);
            auto func = std::get<0>(tup);
            auto mat_func = std::get<1>(tup);
            auto name = std::get<2>(tup);

            func(control, target, state, dim);
            test_state = mat_func(control,target,n) * test_state;
            state_equal(state, test_state, dim, name);
        }
    }
    release_quantum_state(state);
}


TEST(UpdateTest, SingleQubitPauliTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    UINT target, pauli;
    double angle;

    Eigen::MatrixXcd Identity(2, 2);
    Identity << 1, 0, 0, 1;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        /* single qubit Pauli gate */
        target = rand_int(n);
        pauli = rand_int(4);
        single_qubit_Pauli_gate(target, pauli, state, dim);
        test_state = get_expanded_eigen_matrix_with_identity(target, get_eigen_matrix_single_Pauli(pauli), n) * test_state;
        state_equal(state, test_state, dim, "single Pauli gate");

        target = rand_int(n);
        pauli = rand_int(4);
        angle = rand_real();
        single_qubit_Pauli_rotation_gate(target, pauli, angle, state, dim);
        test_state = get_expanded_eigen_matrix_with_identity(target, cos(angle/2)*Identity + 1.i * sin(angle/2) * get_eigen_matrix_single_Pauli(pauli), n) * test_state;
        state_equal(state, test_state, dim, "single rotation Pauli gate");
    }
    release_quantum_state(state);
}



TEST(UpdateTest, MultiQubitPauliTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    UINT pauli;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // multi pauli whole
        std::vector<UINT> pauli_whole, pauli_partial, pauli_partial_index;
        std::vector<std::pair<UINT, UINT>> pauli_partial_pair;

        pauli_whole.resize(n);
        for (UINT i = 0; i < n; ++i) {
            pauli_whole[i] = rand_int(4);
        }
        multi_qubit_Pauli_gate_whole_list(pauli_whole.data(), n, state, dim);
        test_state = get_eigen_matrix_full_qubit_pauli(pauli_whole) * test_state;
        state_equal(state, test_state, dim, "multi Pauli whole gate");

        // multi pauli partial
        pauli_partial.clear();
        pauli_partial_index.clear();
        pauli_partial_pair.clear();
        for (UINT i = 0; i < n; ++i) {
            pauli = rand_int(4);
            pauli_whole[i] = pauli;
            if (pauli != 0) {
                pauli_partial_pair.push_back(std::make_pair(i, pauli));
            }
        }
        std::random_shuffle(pauli_partial_pair.begin(), pauli_partial_pair.end());
        for (auto val : pauli_partial_pair) {
            pauli_partial_index.push_back(val.first);
            pauli_partial.push_back(val.second);
        }
        multi_qubit_Pauli_gate_partial_list(pauli_partial_index.data(), pauli_partial.data(), (UINT)pauli_partial.size(), state, dim);
        test_state = get_eigen_matrix_full_qubit_pauli(pauli_whole) * test_state;
        state_equal(state, test_state, dim, "multi Pauli partial gate");
    }
    release_quantum_state(state);
}

TEST(UpdateTest, MultiQubitPauliRotationTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    UINT pauli;
    double angle;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {

        std::vector<UINT> pauli_whole, pauli_partial, pauli_partial_index;
        std::vector<std::pair<UINT, UINT>> pauli_partial_pair;

        // multi pauli rotation whole
        pauli_whole.resize(n);
        for (UINT i = 0; i < n; ++i) {
            pauli_whole[i] = rand_int(4);
        }
        angle = rand_real();
        multi_qubit_Pauli_rotation_gate_whole_list(pauli_whole.data(), n, angle, state, dim);
        test_state = (cos(angle/2)*whole_I + 1.i * sin(angle/2)* get_eigen_matrix_full_qubit_pauli(pauli_whole)) * test_state;
        state_equal(state, test_state, dim, "multi Pauli rotation whole gate");

        // multi pauli rotation partial
        pauli_partial.clear();
        pauli_partial_index.clear();
        pauli_partial_pair.clear();
        for (UINT i = 0; i < n; ++i) {
            pauli = rand_int(4);
            pauli_whole[i] = pauli;
            if (pauli != 0) {
                pauli_partial_pair.push_back(std::make_pair(i, pauli));
            }
        }
        std::random_shuffle(pauli_partial_pair.begin(), pauli_partial_pair.end());
        for (auto val : pauli_partial_pair) {
            pauli_partial_index.push_back(val.first);
            pauli_partial.push_back(val.second);
        }
        angle = rand_real();
        multi_qubit_Pauli_rotation_gate_partial_list(pauli_partial_index.data(), pauli_partial.data(), (UINT)pauli_partial.size(), angle, state, dim);
        test_state = (cos(angle/2)*whole_I + 1.i * sin(angle/2)* get_eigen_matrix_full_qubit_pauli(pauli_whole)) * test_state;
        state_equal(state, test_state, dim, "multi Pauli rotation partial gate");
    }
    release_quantum_state(state);
}




TEST(UpdateTest, SingleDenseMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

    UINT target;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // single qubit dense matrix gate
        // NOTE: Eigen uses column major by default. To use raw-data of eigen matrix, we need to specify RowMajor.
        target = rand_int(n);
        U = get_eigen_matrix_random_single_qubit_unitary();
        single_qubit_dense_matrix_gate(target, (CTYPE*)U.data(), state, dim);
        test_state = get_expanded_eigen_matrix_with_identity(target, U, n) * test_state;
        state_equal(state, test_state, dim, "single dense gate");
    }
    release_quantum_state(state);
}

TEST(UpdateTest, SingleDiagonalMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    Eigen::MatrixXcd Identity(2, 2),Z(2, 2);
    Identity << 1, 0, 0, 1;
    Z << 1, 0, 0, -1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

    UINT target;
    double icoef, zcoef, norm;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // single qubit diagonal matrix gate
        target = rand_int(n);
        icoef = rand_real(); zcoef = rand_real();
        norm = sqrt(icoef * icoef + zcoef * zcoef);
        icoef /= norm; zcoef /= norm;
        U = icoef * Identity + 1.i*zcoef * Z;
        Eigen::VectorXcd diag = U.diagonal();
        single_qubit_diagonal_matrix_gate(target, (CTYPE*)diag.data(), state, dim);
        test_state = get_expanded_eigen_matrix_with_identity(target, U, n) * test_state;
        state_equal(state, test_state, dim, "single diagonal gate");
    }
    release_quantum_state(state);
}



TEST(UpdateTest, SinglePhaseMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

    UINT target;
    double angle;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // single qubit phase matrix gate
        target = rand_int(n);
        angle = rand_real();
        U << 1, 0, 0, cos(angle) + 1.i*sin(angle);
        single_qubit_phase_gate(target, cos(angle) + 1.i*sin(angle), state, dim);
        test_state = get_expanded_eigen_matrix_with_identity(target, U, n) * test_state;
        state_equal(state, test_state, dim, "single phase gate");
    }
    release_quantum_state(state);
}

TEST(UpdateTest, SingleQubitControlSingleQubitDenseMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

    UINT target,control;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // single qubit control-1 single qubit gate
        target = rand_int(n);
        control = rand_int(n - 1);
        if (control >= target) control++;
        U = get_eigen_matrix_random_single_qubit_unitary();
        single_qubit_control_single_qubit_dense_matrix_gate(control, 1, target, (CTYPE*)U.data(), state, dim);
        test_state = (get_expanded_eigen_matrix_with_identity(control, P0, n) + get_expanded_eigen_matrix_with_identity(control, P1, n)*get_expanded_eigen_matrix_with_identity(target, U, n)) * test_state;
        state_equal(state, test_state, dim, "single qubit control sinlge qubit dense gate");

        // single qubit control-0 single qubit gate
        target = rand_int(n);
        control = rand_int(n - 1);
        if (control >= target) control++;
        U = get_eigen_matrix_random_single_qubit_unitary();
        single_qubit_control_single_qubit_dense_matrix_gate(control, 0, target, (CTYPE*)U.data(), state, dim);
        test_state = (get_expanded_eigen_matrix_with_identity(control, P1, n) + get_expanded_eigen_matrix_with_identity(control, P0, n)*get_expanded_eigen_matrix_with_identity(target, U, n)) * test_state;
        state_equal(state, test_state, dim, "single qubit control sinlge qubit dense gate");
    }
    release_quantum_state(state);
}



TEST(UpdateTest, TwoQubitControlSingleQubitDenseMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

    UINT target;
    UINT controls[2];

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // two qubit control-10 single qubit gate
        std::random_shuffle(index_list.begin(), index_list.end());
        target = index_list[0];
        controls[0] = index_list[1];
        controls[1] = index_list[2];

        U = get_eigen_matrix_random_single_qubit_unitary();
        UINT mvalues[2] = { 1,0 };
        multi_qubit_control_single_qubit_dense_matrix_gate(controls, mvalues, 2, target, (CTYPE*)U.data(), state, dim);
        test_state = (
            get_expanded_eigen_matrix_with_identity(controls[0], P0, n)*get_expanded_eigen_matrix_with_identity(controls[1], P0, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P0, n)*get_expanded_eigen_matrix_with_identity(controls[1], P1, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P1, n)*get_expanded_eigen_matrix_with_identity(controls[1], P0, n)*get_expanded_eigen_matrix_with_identity(target, U, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P1, n)*get_expanded_eigen_matrix_with_identity(controls[1], P1, n)
            ) * test_state;
        state_equal(state, test_state, dim, "two qubit control sinlge qubit dense gate");

    }
    release_quantum_state(state);
}


TEST(UpdateTest, TwoQubitDenseMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U,U2;
    Eigen::Matrix<std::complex<double>, 4,4, Eigen::RowMajor> Umerge;

    UINT targets[2];

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {

        // two qubit dense matrix gate
        U = get_eigen_matrix_random_single_qubit_unitary();
        U2 = get_eigen_matrix_random_single_qubit_unitary();

        std::random_shuffle(index_list.begin(), index_list.end());

        targets[0] = index_list[0];
        targets[1] = index_list[1];
        Umerge = kronecker_product(U2, U);
        // the below two lines are equivalent to the above two line
        //UINT targets_rev[2] = { targets[1], targets[0] };
        //Umerge = kronecker_product(U, U2);
        test_state = get_expanded_eigen_matrix_with_identity(targets[1], U2, n) * get_expanded_eigen_matrix_with_identity(targets[0], U, n) * test_state;
        multi_qubit_dense_matrix_gate(targets, 2, (CTYPE*)Umerge.data(), state, dim);
        state_equal(state, test_state, dim, "two-qubit separable dense gate");
    }
    release_quantum_state(state);
}



TEST(UpdateTest, SingleQubitControlTwoQubitDenseMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U, U2;
    Eigen::Matrix<std::complex<double>, 4, 4, Eigen::RowMajor> Umerge;

    UINT targets[2], control;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // single qubit 1-controlled qubit dense matrix gate
        U = get_eigen_matrix_random_single_qubit_unitary();
        U2 = get_eigen_matrix_random_single_qubit_unitary();
        std::random_shuffle(index_list.begin(), index_list.end());
        targets[0] = index_list[0];
        targets[1] = index_list[1];
        control = index_list[2];

        Umerge = kronecker_product(U2, U);
        test_state = (get_expanded_eigen_matrix_with_identity(control, P0, n) + get_expanded_eigen_matrix_with_identity(control, P1, n)*get_expanded_eigen_matrix_with_identity(targets[1], U2, n) * get_expanded_eigen_matrix_with_identity(targets[0], U, n)) * test_state;
        single_qubit_control_multi_qubit_dense_matrix_gate(control, 1, targets, 2, (CTYPE*)Umerge.data(), state, dim);
        state_equal(state, test_state, dim, "single qubit control two-qubit separable dense gate");
    }
    release_quantum_state(state);
}



TEST(UpdateTest, TwoQubitControlTwoQubitDenseMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U, U2;
    Eigen::Matrix<std::complex<double>, 4, 4, Eigen::RowMajor> Umerge;

    UINT targets[2], controls[2],mvalues[2];

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {

        // two qubit control-11 two qubit gate
        U = get_eigen_matrix_random_single_qubit_unitary();
        U2 = get_eigen_matrix_random_single_qubit_unitary();
        std::random_shuffle(index_list.begin(), index_list.end());
        targets[0] = index_list[0];
        targets[1] = index_list[1];
        controls[0] = index_list[2];
        controls[1] = index_list[3];

        mvalues[0] = 1; mvalues[1] = 1;
        Umerge = kronecker_product(U2, U);
        multi_qubit_control_multi_qubit_dense_matrix_gate(controls, mvalues, 2, targets, 2, (CTYPE*)Umerge.data(), state, dim);
        test_state = (
            get_expanded_eigen_matrix_with_identity(controls[0], P0, n)*get_expanded_eigen_matrix_with_identity(controls[1], P0, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P0, n)*get_expanded_eigen_matrix_with_identity(controls[1], P1, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P1, n)*get_expanded_eigen_matrix_with_identity(controls[1], P0, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P1, n)*get_expanded_eigen_matrix_with_identity(controls[1], P1, n)*get_expanded_eigen_matrix_with_identity(targets[0], U, n)*get_expanded_eigen_matrix_with_identity(targets[1], U2, n)
            ) * test_state;
        state_equal(state, test_state, dim, "two qubit control two qubit dense gate");
    }
    release_quantum_state(state);
}
