#include <gtest/gtest.h>
#include "../util/util.h"
#include <cppsim/state.hpp>
#include <cppsim/utility.hpp>
#include <cppsim/state_gpu.hpp>

// post-selection probability check
// referred to test/csim/test_stat.cpp
TEST(StatOperationTest, ProbTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	//CTYPE* state = allocate_quantum_state(dim);
	QuantumStateGpu state(n);
	Eigen::MatrixXcd P0(2, 2), P1(2, 2);
	P0 << 1, 0, 0, 0;
	P1 << 0, 0, 0, 1;

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		state.set_Haar_random_state();
		ASSERT_NEAR(state_norm_host(state.data(), dim), 1, eps);
		auto state_cpp = state.data_cpp();
		Eigen::VectorXcd test_state(dim);
		for (ITYPE i = 0; i < dim; ++i) test_state[i] = state_cpp[i];

		for (UINT target = 0; target < n; ++target) {
			double p0 = M0_prob_host(target, state.data(), dim);
			double p1 = M1_prob_host(target, state.data(), dim);
			ASSERT_NEAR((get_expanded_eigen_matrix_with_identity(target, P0, n)*test_state).squaredNorm(), p0, eps);
			ASSERT_NEAR((get_expanded_eigen_matrix_with_identity(target, P1, n)*test_state).squaredNorm(), p1, eps);
			ASSERT_NEAR(p0 + p1, 1, eps);
		}
	}
}

// marginal probability check
// referred to test/csim/test_stat.cpp
TEST(StatOperationTest, MarginalProbTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	QuantumStateGpu state(n);
	Eigen::MatrixXcd P0(2, 2), P1(2, 2), Identity(2, 2);
	P0 << 1, 0, 0, 0;
	P1 << 0, 0, 0, 1;
	Identity << 1, 0, 0, 1;

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		state.set_Haar_random_state();
		ASSERT_NEAR(state_norm_host(state.data(), dim), 1, eps);
		auto state_cpp = state.data_cpp();
		Eigen::VectorXcd test_state(dim);
		for (ITYPE i = 0; i < dim; ++i) test_state[i] = state_cpp[i];

		for (UINT target = 0; target < n; ++target) {
			// merginal probability check
			Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
			std::vector<UINT> index_list, measured_value_list;

			index_list.clear();
			measured_value_list.clear();
			for (UINT i = 0; i < n; ++i) {
				UINT measured_value = rand_int(3);
				if (measured_value != 2) {
					measured_value_list.push_back(measured_value);
					index_list.push_back(i);
				}
				if (measured_value == 0) {
					mat = kronecker_product(P0, mat);
				}
				else if (measured_value == 1) {
					mat = kronecker_product(P1, mat);
				}
				else {
					mat = kronecker_product(Identity, mat);
				}
			}
			double test_marginal_prob = (mat*test_state).squaredNorm();
			double res = marginal_prob_host(index_list.data(), measured_value_list.data(), (UINT)index_list.size(), state.data(), dim);
			ASSERT_NEAR(test_marginal_prob, res, eps);
		}
	}
}

TEST(StateTest, GenerateAndRelease) {    
    UINT n = 10;
    double eps = 1e-14;
    QuantumState state(n);
    ASSERT_EQ(state.qubit_count, n);
    ASSERT_EQ(state.dim, 1ULL << n);
    state.set_zero_state();
    for (UINT i = 0; i < state.dim; ++i) {
        if (i == 0) ASSERT_NEAR(abs(state.data_cpp()[i] - 1.), 0, eps);
        else ASSERT_NEAR(abs(state.data_cpp()[i]), 0, eps);
    }
    Random random;
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        ITYPE basis = random.int64()%state.dim;
        state.set_computational_basis(basis);
        for (UINT i = 0; i < state.dim; ++i) {
            if (i == basis) ASSERT_NEAR(abs(state.data_cpp()[i] - 1.), 0, eps);
            else ASSERT_NEAR(abs(state.data_cpp()[i]), 0, eps);
        }
    }
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        state.set_Haar_random_state();
        ASSERT_NEAR(state.get_norm(),1.,eps);
    }
}

TEST(StateTest, Sampling) {
    UINT n = 10;
    QuantumState state(n);
    state.set_Haar_random_state();
    state.set_computational_basis(100);
    auto res1 = state.sampling(1024);
    state.set_computational_basis(100);
    auto res2 = state.sampling(1024);
}


TEST(StateTest, SetState) {
	const double eps = 1e-10;
	const UINT n = 10;
	QuantumStateGpu state(n);
	const ITYPE dim = 1ULL << n;
	std::vector<std::complex<double>> state_vector(dim);
	for (UINT i = 0; i < dim; ++i) {
		double d = (double)i;
		state_vector[i] = d + std::complex<double>(0, 1)*(d + 0.1);
	}
	state.load(state_vector);
	for (UINT i = 0; i < dim; ++i) {
		ASSERT_NEAR(state.data_cpp()[i].real(), state_vector[i].real(), eps);
		ASSERT_NEAR(state.data_cpp()[i].imag(), state_vector[i].imag(), eps);
	}
}

TEST(StateTest, AddState) {
	const double eps = 1e-10;
	const UINT n = 10;
	QuantumStateGpu state1(n);
	QuantumStateGpu state2(n);
	state1.set_Haar_random_state();
	state2.set_Haar_random_state();

	const ITYPE dim = 1ULL << n;
	std::vector<std::complex<double>> state_vector1(dim);
	std::vector<std::complex<double>> state_vector2(dim);
	for (ITYPE i = 0; i < dim; ++i) {
		state_vector1[i] = state1.data_cpp()[i];
		state_vector2[i] = state2.data_cpp()[i];
	}

	state1.add_state(&state2);

	for (ITYPE i = 0; i < dim; ++i) {
		ASSERT_NEAR(state1.data_cpp()[i].real(), state_vector1[i].real() + state_vector2[i].real(), eps);
		ASSERT_NEAR(state1.data_cpp()[i].imag(), state_vector1[i].imag() + state_vector2[i].imag(), eps);
		ASSERT_NEAR(state2.data_cpp()[i].real(), state_vector2[i].real(), eps);
		ASSERT_NEAR(state2.data_cpp()[i].imag(), state_vector2[i].imag(), eps);
	}
}

TEST(StateTest, MultiplyCoef) {
	const double eps = 1e-10;
	const UINT n = 10;
	const std::complex<double> coef(0.5, 0.2);

	QuantumStateGpu state(n);
	state.set_Haar_random_state();

	const ITYPE dim = 1ULL << n;
	std::vector<std::complex<double>> state_vector(dim);
	for (ITYPE i = 0; i < dim; ++i) {
		state_vector[i] = state.data_cpp()[i] * coef;
	}
	state.multiply_coef(coef);

	for (ITYPE i = 0; i < dim; ++i) {
		ASSERT_NEAR(state.data_cpp()[i].real(), state_vector[i].real(), eps);
		ASSERT_NEAR(state.data_cpp()[i].imag(), state_vector[i].imag(), eps);
	}
}
