#include <gtest/gtest.h>
#include "../util/util.h"
#include <cppsim/state.hpp>
#include <cppsim/utility.hpp>

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
        ASSERT_NEAR(state.get_squared_norm(),1.,eps);
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
	QuantumState state(n);
	const ITYPE dim = 1ULL << n;
	std::vector<std::complex<double>> state_vector(dim);
	for (ITYPE i = 0; i < dim; ++i) {
		double d = (double)i;
		state_vector[i] = d + std::complex<double>(0, 1)*(d + 0.1);
	}
	state.load(state_vector);
	for (ITYPE i = 0; i < dim; ++i) {
		ASSERT_NEAR(state.data_cpp()[i].real(), state_vector[i].real(), eps);
		ASSERT_NEAR(state.data_cpp()[i].imag(), state_vector[i].imag(), eps);
	}
}

TEST(StateTest, GetMarginalProbability) {
	const double eps = 1e-10;
	const UINT n = 2;
	const ITYPE dim = 1 << n;
	QuantumState state(n);
	state.set_Haar_random_state();
	std::vector<double> probs;
	for (ITYPE i = 0; i < dim; ++i) {
		probs.push_back(pow(abs(state.data_cpp()[i]),2));
	}
	ASSERT_NEAR(state.get_marginal_probability({ 0,0 }), probs[0], eps);
	ASSERT_NEAR(state.get_marginal_probability({ 1,0 }), probs[1], eps);
	ASSERT_NEAR(state.get_marginal_probability({ 0,1 }), probs[2], eps);
	ASSERT_NEAR(state.get_marginal_probability({ 1,1 }), probs[3], eps);
	ASSERT_NEAR(state.get_marginal_probability({ 0,2 }), probs[0] + probs[2], eps);
	ASSERT_NEAR(state.get_marginal_probability({ 1,2 }), probs[1] + probs[3], eps);
	ASSERT_NEAR(state.get_marginal_probability({ 2,0 }), probs[0] + probs[1], eps);
	ASSERT_NEAR(state.get_marginal_probability({ 2,1 }), probs[2] + probs[3], eps);
	ASSERT_NEAR(state.get_marginal_probability({ 2,2 }), 1., eps);
}


TEST(StateTest, AddState) {
	const double eps = 1e-10;
	const UINT n = 10;
	QuantumState state1(n);
	QuantumState state2(n);
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

	QuantumState state(n);
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