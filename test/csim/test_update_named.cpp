
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <csim/init_ops.hpp>
#include <csim/memory_ops.hpp>
#include <csim/stat_ops.hpp>
#include <csim/update_ops.hpp>
#include <csim/update_ops_cpp.hpp>
#include <string>

#include "../util/util.hpp"

void test_single_qubit_named_gate(UINT n, std::string name,
    std::function<void(UINT, CTYPE*, ITYPE)> func, Eigen::MatrixXcd mat) {
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 2;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state_with_seed(state, dim, 0);

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = (std::complex<double>)state[i];
    std::vector<UINT> indices;
    for (UINT i = 0; i < n; ++i) indices.push_back(i);

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        for (UINT i = 0; i < n; ++i) {
            UINT target = indices[i];
            func(target, state, dim);
            test_state =
                get_expanded_eigen_matrix_with_identity(target, mat, n) *
                test_state;
            state_equal(state, test_state, dim, name);
        }
        std::shuffle(indices.begin(), indices.end(), engine);
    }
    release_quantum_state(state);
}

TEST(UpdateTest, XGate) {
    Eigen::MatrixXcd mat(2, 2);
    mat << 0, 1, 1, 0;
    test_single_qubit_named_gate(6, "XGate", X_gate, mat);
    test_single_qubit_named_gate(6, "XGate", X_gate_parallel_unroll, mat);
#ifdef _USE_SIMD
    test_single_qubit_named_gate(6, "XGate", X_gate_parallel_simd, mat);
#endif
#ifdef _USE_SVE
    test_single_qubit_named_gate(
        1, "XGate", X_gate_parallel_sve, mat);  //  256-bit
    test_single_qubit_named_gate(
        2, "XGate", X_gate_parallel_sve, mat);  //  512-bit
    test_single_qubit_named_gate(
        3, "XGate", X_gate_parallel_sve, mat);  // 1024-bit
    test_single_qubit_named_gate(
        4, "XGate", X_gate_parallel_sve, mat);  // 2048-bit
    test_single_qubit_named_gate(6, "XGate", X_gate_parallel_sve, mat);
#endif
}
TEST(UpdateTest, YGate) {
    Eigen::MatrixXcd mat(2, 2);
    mat << 0, -1.i, 1.i, 0;
    test_single_qubit_named_gate(6, "YGate", Y_gate, mat);
    test_single_qubit_named_gate(6, "YGate", Y_gate_parallel_unroll, mat);
#ifdef _USE_SIMD
    test_single_qubit_named_gate(6, "YGate", Y_gate_parallel_simd, mat);
#endif
#ifdef _USE_SVE
    test_single_qubit_named_gate(
        1, "YGate", Y_gate_parallel_sve, mat);  //  256-bit
    test_single_qubit_named_gate(
        2, "YGate", Y_gate_parallel_sve, mat);  //  512-bit
    test_single_qubit_named_gate(
        3, "YGate", Y_gate_parallel_sve, mat);  // 1024-bit
    test_single_qubit_named_gate(
        4, "YGate", Y_gate_parallel_sve, mat);  // 2048-bit
    test_single_qubit_named_gate(6, "YGate", Y_gate_parallel_sve, mat);
#endif
}
TEST(UpdateTest, ZGate) {
    const UINT n = 3;
    Eigen::MatrixXcd mat(2, 2);
    mat << 1, 0, 0, -1;
    test_single_qubit_named_gate(6, "ZGate", Z_gate, mat);
    test_single_qubit_named_gate(6, "ZGate", Z_gate_parallel_unroll, mat);
#ifdef _USE_SIMD
    test_single_qubit_named_gate(6, "ZGate", Z_gate_parallel_simd, mat);
#endif
#ifdef _USE_SVE
    test_single_qubit_named_gate(
        1, "ZGate", Z_gate_parallel_sve, mat);  //  256-bit
    test_single_qubit_named_gate(
        2, "ZGate", Z_gate_parallel_sve, mat);  //  512-bit
    test_single_qubit_named_gate(
        3, "ZGate", Z_gate_parallel_sve, mat);  // 1024-bit
    test_single_qubit_named_gate(
        4, "ZGate", Z_gate_parallel_sve, mat);  // 2048-bit
    test_single_qubit_named_gate(6, "ZGate", Z_gate_parallel_sve, mat);
#endif
}
TEST(UpdateTest, HGate) {
    const UINT n = 3;
    Eigen::MatrixXcd mat(2, 2);
    mat << 1, 1, 1, -1;
    mat /= sqrt(2.);
    test_single_qubit_named_gate(n, "HGate", H_gate, mat);
    test_single_qubit_named_gate(6, "HGate", H_gate_parallel_unroll, mat);
#ifdef _USE_SIMD
    test_single_qubit_named_gate(6, "HGate", H_gate_parallel_simd, mat);
#endif
#ifdef _USE_SVE
    test_single_qubit_named_gate(
        1, "HGate", H_gate_parallel_sve, mat);  //  256-bit
    test_single_qubit_named_gate(
        2, "HGate", H_gate_parallel_sve, mat);  //  512-bit
    test_single_qubit_named_gate(
        3, "HGate", H_gate_parallel_sve, mat);  // 1024-bit
    test_single_qubit_named_gate(
        4, "HGate", H_gate_parallel_sve, mat);  // 2048-bit
    test_single_qubit_named_gate(6, "HGate", H_gate_parallel_sve, mat);
#endif
}

TEST(UpdateTest, SGate) {
    const UINT n = 3;
    Eigen::MatrixXcd mat(2, 2);
    mat << 1, 0, 0, 1.i;
    test_single_qubit_named_gate(n, "SGate", S_gate, mat);
    test_single_qubit_named_gate(n, "SGate", Sdag_gate, mat.adjoint());
}

TEST(UpdateTest, TGate) {
    const UINT n = 3;
    Eigen::MatrixXcd mat(2, 2);
    mat << 1, 0, 0, (1. + 1.i) / sqrt(2.);
    test_single_qubit_named_gate(n, "TGate", T_gate, mat);
    test_single_qubit_named_gate(n, "TGate", Tdag_gate, mat.adjoint());
}

TEST(UpdateTest, sqrtXGate) {
    const UINT n = 3;
    Eigen::MatrixXcd mat(2, 2);
    mat << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
    test_single_qubit_named_gate(n, "SqrtXGate", sqrtX_gate, mat);
    test_single_qubit_named_gate(
        n, "SqrtXdagGate", sqrtXdag_gate, mat.adjoint());
}

TEST(UpdateTest, sqrtYGate) {
    const UINT n = 3;
    Eigen::MatrixXcd mat(2, 2);
    mat << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
    test_single_qubit_named_gate(n, "SqrtYGate", sqrtY_gate, mat);
    test_single_qubit_named_gate(
        n, "SqrtYdagGate", sqrtYdag_gate, mat.adjoint());
}

void test_projection_gate(std::function<void(UINT, CTYPE*, ITYPE)> func,
    std::function<double(UINT, CTYPE*, ITYPE)> prob_func,
    Eigen::MatrixXcd mat) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    UINT target;
    double prob;

    auto state = allocate_quantum_state(dim);
    std::vector<UINT> indices;
    for (UINT i = 0; i < n; ++i) indices.push_back(i);

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        for (int i = 0; i < n; ++i) {
            target = indices[i];
            initialize_Haar_random_state(state, dim);
            Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
            for (ITYPE i = 0; i < dim; ++i)
                test_state[i] = (std::complex<double>)state[i];

            // Z-projection operators
            prob = prob_func(target, state, dim);
            EXPECT_GT(prob, 1e-10);
            func(target, state, dim);
            ASSERT_NEAR(state_norm_squared(state, dim), prob, eps);
            normalize(prob, state, dim);

            test_state =
                get_expanded_eigen_matrix_with_identity(target, mat, n) *
                test_state;
            ASSERT_NEAR(test_state.squaredNorm(), prob, eps);
            test_state.normalize();
            state_equal(state, test_state, dim, "Projection gate");
        }
        std::shuffle(indices.begin(), indices.end(), engine);
    }
    release_quantum_state(state);
}

TEST(UpdateTest, ProjectionAndNormalizeTest) {
    const auto P0 = make_P0();
    const auto P1 = make_P1();
    test_projection_gate(P0_gate, M0_prob, P0);
    test_projection_gate(P1_gate, M1_prob, P1);
    test_projection_gate(P0_gate_parallel, M0_prob, P0);
    test_projection_gate(P1_gate_parallel, M1_prob, P1);
}

TEST(UpdateTest, SingleQubitRotationGateTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();

    UINT target;
    double angle;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = (std::complex<double>)state[i];
    typedef std::tuple<std::function<void(UINT, double, CTYPE*, ITYPE)>,
        Eigen::MatrixXcd, std::string>
        testset;
    std::vector<testset> test_list;
    test_list.push_back(std::make_tuple(RX_gate, X, "Xrot"));
    test_list.push_back(std::make_tuple(RY_gate, Y, "Yrot"));
    test_list.push_back(std::make_tuple(RZ_gate, Z, "Zrot"));
    std::complex<double> imag_unit(0, 1);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        for (auto tup : test_list) {
            target = rand_int(n);
            angle = rand_real();
            auto func = std::get<0>(tup);
            auto mat = std::get<1>(tup);
            auto name = std::get<2>(tup);
            func(target, angle, state, dim);
            test_state = get_expanded_eigen_matrix_with_identity(target,
                             cos(angle / 2) * Identity +
                                 imag_unit * sin(angle / 2) * mat,
                             n) *
                         test_state;
            state_equal(state, test_state, dim, name);
        }
    }
    release_quantum_state(state);
}

void test_two_qubit_named_gate(UINT n, std::string name,
    std::function<void(UINT, UINT, CTYPE*, ITYPE)> func,
    std::function<Eigen::MatrixXcd(UINT, UINT, UINT)> matfunc) {
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 2;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state_with_seed(state, dim, 0);

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = (std::complex<double>)state[i];
    std::vector<UINT> indices;
    for (UINT i = 0; i < n; ++i) indices.push_back(i);

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        for (UINT i = 0; i + 1 < n; i += 2) {
            UINT target = indices[i];
            UINT control = indices[i + 1];
            func(control, target, state, dim);
            Eigen::MatrixXcd mat = matfunc(control, target, n);
            test_state = mat * test_state;
            state_equal(state, test_state, dim, name);
        }
        std::shuffle(indices.begin(), indices.end(), engine);
    }
    release_quantum_state(state);
}

TEST(UpdateTest, CNOTGate) {
    const UINT n = 4;
    test_two_qubit_named_gate(
        n, "CNOT", CNOT_gate, get_eigen_matrix_full_qubit_CNOT);
    test_two_qubit_named_gate(6, "CNOTGate", CNOT_gate_parallel_unroll,
        get_eigen_matrix_full_qubit_CNOT);
#ifdef _USE_SIMD
    test_two_qubit_named_gate(6, "CNOTGate", CNOT_gate_parallel_simd,
        get_eigen_matrix_full_qubit_CNOT);
#endif
#ifdef _USE_SVE
    test_two_qubit_named_gate(1, "CNOTGate", CNOT_gate_parallel_sve,
        get_eigen_matrix_full_qubit_CNOT);  //  256-bit
    test_two_qubit_named_gate(2, "CNOTGate", CNOT_gate_parallel_sve,
        get_eigen_matrix_full_qubit_CNOT);  //  512-bit
    test_two_qubit_named_gate(3, "CNOTGate", CNOT_gate_parallel_sve,
        get_eigen_matrix_full_qubit_CNOT);  // 1024-bit
    test_two_qubit_named_gate(4, "CNOTGate", CNOT_gate_parallel_sve,
        get_eigen_matrix_full_qubit_CNOT);  // 2048-bit
    test_two_qubit_named_gate(6, "CNOTGate", CNOT_gate_parallel_sve,
        get_eigen_matrix_full_qubit_CNOT);
#endif
}

TEST(UpdateTest, CZGate) {
    const UINT n = 4;
    test_two_qubit_named_gate(n, "CZ", CZ_gate, get_eigen_matrix_full_qubit_CZ);
    test_two_qubit_named_gate(
        6, "CZGate", CZ_gate_parallel_unroll, get_eigen_matrix_full_qubit_CZ);
#ifdef _USE_SIMD
    test_two_qubit_named_gate(
        6, "CZGate", CZ_gate_parallel_simd, get_eigen_matrix_full_qubit_CZ);
#endif
#ifdef _USE_SVE
    test_two_qubit_named_gate(1, "CZGate", CZ_gate_parallel_sve,
        get_eigen_matrix_full_qubit_CZ);  //  256-bit
    test_two_qubit_named_gate(2, "CZGate", CZ_gate_parallel_sve,
        get_eigen_matrix_full_qubit_CZ);  //  512-bit
    test_two_qubit_named_gate(3, "CZGate", CZ_gate_parallel_sve,
        get_eigen_matrix_full_qubit_CZ);  // 1024-bit
    test_two_qubit_named_gate(4, "CZGate", CZ_gate_parallel_sve,
        get_eigen_matrix_full_qubit_CZ);  // 2048-bit
    test_two_qubit_named_gate(
        6, "CZGate", CZ_gate_parallel_sve, get_eigen_matrix_full_qubit_CZ);
#endif
}

TEST(UpdateTest, SWAPGate) {
    const UINT n = 4;
    test_two_qubit_named_gate(
        n, "SWAP", SWAP_gate, get_eigen_matrix_full_qubit_SWAP);
    test_two_qubit_named_gate(6, "SWAPGate", SWAP_gate_parallel_unroll,
        get_eigen_matrix_full_qubit_SWAP);
#ifdef _USE_SIMD
    test_two_qubit_named_gate(6, "SWAPGate", SWAP_gate_parallel_simd,
        get_eigen_matrix_full_qubit_SWAP);
#endif
#ifdef _USE_SVE
    test_two_qubit_named_gate(1, "SWAPGate", SWAP_gate_parallel_sve,
        get_eigen_matrix_full_qubit_SWAP);  //  256-bit
    test_two_qubit_named_gate(2, "SWAPGate", SWAP_gate_parallel_sve,
        get_eigen_matrix_full_qubit_SWAP);  //  512-bit
    test_two_qubit_named_gate(3, "SWAPGate", SWAP_gate_parallel_sve,
        get_eigen_matrix_full_qubit_SWAP);  // 1024-bit
    test_two_qubit_named_gate(4, "SWAPGate", SWAP_gate_parallel_sve,
        get_eigen_matrix_full_qubit_SWAP);  // 2048-bit
    test_two_qubit_named_gate(6, "SWAPGate", SWAP_gate_parallel_sve,
        get_eigen_matrix_full_qubit_SWAP);
#endif
}
