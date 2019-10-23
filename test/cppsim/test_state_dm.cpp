#include <gtest/gtest.h>
#include "../util/util.h"
#include <cppsim/state_dm.hpp>
#include <cppsim/utility.hpp>

TEST(DensityMatrixTest, GenerateAndRelease) {
	UINT n = 5;
	double eps = 1e-14;
	const ITYPE dim = 1ULL << n;
	DensityMatrix state(n);
	ASSERT_EQ(state.qubit_count, n);
	ASSERT_EQ(state.dim, dim);
	state.set_zero_state();
	for (UINT i = 0; i < state.dim; ++i) {
		for (UINT j = 0; j < state.dim; ++j) {
			if (i == 0 && j == 0) ASSERT_NEAR(abs(state.data_cpp()[i*dim+j] - 1.), 0, eps);
			else ASSERT_NEAR(abs(state.data_cpp()[i*dim+j]), 0, eps);
		}
	}
	Random random;
	for (UINT repeat = 0; repeat < 10; ++repeat) {
		ITYPE basis = random.int64() % state.dim;
		state.set_computational_basis(basis);
		for (UINT i = 0; i < state.dim; ++i) {
			for (UINT j = 0; j < state.dim; ++j) {
				if (i == basis && j == basis) ASSERT_NEAR(abs(state.data_cpp()[i*dim+j] - 1.), 0, eps);
				else ASSERT_NEAR(abs(state.data_cpp()[i*dim+j]), 0, eps);
			}
		}
	}
	for (UINT repeat = 0; repeat < 10; ++repeat) {
		state.set_Haar_random_state();
		ASSERT_NEAR(state.get_squared_norm(), 1., eps);
	}
}

TEST(DensityMatrixTest, Sampling) {
	UINT n = 5;
	DensityMatrix state(n);
	state.set_Haar_random_state();
	state.set_computational_basis(10);
	auto res1 = state.sampling(1024);
	state.set_computational_basis(10);
	auto res2 = state.sampling(1024);
}


TEST(DensityMatrixTest, SetState) {
	const double eps = 1e-10;
	const UINT n = 5;
	DensityMatrix state(n);
	const ITYPE dim = 1ULL << n;
	std::vector<std::complex<double>> state_vector(dim*dim);
	for (ITYPE i = 0; i < dim; ++i) {
		for (ITYPE j = 0; j < dim; ++j) {
			double d = (double)(i*dim+j);
			state_vector[j*dim + i] = d + std::complex<double>(0, 1)*(d + 0.1);
		}
	}
	state.load(state_vector);
	for (ITYPE i = 0; i < dim; ++i) {
		for(ITYPE j=0;j<dim;++j){
			ASSERT_NEAR(state.data_cpp()[i*dim + j].real(), state_vector[i*dim + j].real(), eps);
			ASSERT_NEAR(state.data_cpp()[i*dim + j].imag(), state_vector[i*dim + j].imag(), eps);
		}
	}
}

TEST(DensityMatrixTest, GetMarginalProbability) {
	const double eps = 1e-10;
	const UINT n = 2;
	const ITYPE dim = 1 << n;
	DensityMatrix state(n);
	state.set_Haar_random_state();
	std::vector<double> probs;
	for (ITYPE i = 0; i < dim; ++i) {
		probs.push_back(real(state.data_cpp()[i*dim+i]));
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


TEST(DensityMatrixTest, AddState) {
	const double eps = 1e-10;
	const UINT n = 5;
	DensityMatrix state1(n);
	DensityMatrix state2(n);
	state1.set_Haar_random_state();
	state2.set_Haar_random_state();

	const ITYPE dim = 1ULL << n;
	std::vector<std::complex<double>> state_vector1(dim*dim);
	std::vector<std::complex<double>> state_vector2(dim*dim);
	for (ITYPE i = 0; i < dim; ++i) {
		for (ITYPE j = 0; j < dim; ++j) {
			state_vector1[i*dim + j] = state1.data_cpp()[i*dim + j];
			state_vector2[i*dim + j] = state2.data_cpp()[i*dim + j];
		}
	}

	state1.add_state(&state2);

	for (ITYPE i = 0; i < dim; ++i) {
		for (ITYPE j = 0; j < dim; ++j) {
			ASSERT_NEAR(state1.data_cpp()[i*dim+j].real(), state_vector1[i*dim + j].real() + state_vector2[i*dim + j].real(), eps);
			ASSERT_NEAR(state1.data_cpp()[i*dim + j].imag(), state_vector1[i*dim + j].imag() + state_vector2[i*dim + j].imag(), eps);
			ASSERT_NEAR(state2.data_cpp()[i*dim + j].real(), state_vector2[i*dim + j].real(), eps);
			ASSERT_NEAR(state2.data_cpp()[i*dim + j].imag(), state_vector2[i*dim + j].imag(), eps);
		}
	}
}

TEST(DensityMatrixTest, MultiplyCoef) {
	const double eps = 1e-10;
	const UINT n = 10;
	const std::complex<double> coef(0.5, 0.2);

	DensityMatrix state(n);
	state.set_Haar_random_state();

	const ITYPE dim = 1ULL << n;
	std::vector<std::complex<double>> state_vector(dim*dim);
	for (ITYPE i = 0; i < dim; ++i) {
		for (ITYPE j = 0; j < dim; ++j) {
			state_vector[i*dim+j] = state.data_cpp()[i*dim+j] * coef;
		}
	}
	state.multiply_coef(coef);

	for (ITYPE i = 0; i < dim; ++i) {
		for (ITYPE j = 0; j < dim; ++j) {
			ASSERT_NEAR(state.data_cpp()[i*dim+j].real(), state_vector[i*dim+j].real(), eps);
			ASSERT_NEAR(state.data_cpp()[i*dim+j].imag(), state_vector[i*dim+j].imag(), eps);
		}
	}
}