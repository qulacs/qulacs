#include <gtest/gtest.h>
#include "../util/util.h"

#include <cmath>
#include <cppsim/state_dm.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/pauli_operator.hpp>

#include <cppsim/utility.hpp>
#include <csim/update_ops.h>
#include <functional>
#include <numeric>


TEST(DensityMatrixGeneralGateTest, ProbabilisticGate) {
	auto gate1 = gate::X(0);
	auto gate2 = gate::X(1);
	auto gate3 = gate::X(2);
	auto prob_gate = gate::Probabilistic({ 0.25,0.25,0.25 }, { gate1, gate2, gate2 });
	DensityMatrix s(3);
	s.set_computational_basis(0);
	prob_gate->update_quantum_state(&s);
	delete gate1;
	delete gate2;
	delete gate3;
	delete prob_gate;
}

TEST(DensityMatrixGeneralGateTest, CPTPGate) {
	auto gate1 = gate::merge(gate::P0(0), gate::P0(1));
	auto gate2 = gate::merge(gate::P0(0), gate::P1(1));
	auto gate3 = gate::merge(gate::P1(0), gate::P0(1));
	auto gate4 = gate::merge(gate::P1(0), gate::P1(1));

	auto CPTP = gate::CPTP({ gate3, gate2, gate1, gate4 });
	DensityMatrix s(3);
	s.set_computational_basis(0);
	CPTP->update_quantum_state(&s);
	s.set_Haar_random_state();
	CPTP->update_quantum_state(&s);
	delete CPTP;
}

TEST(DensityMatrixGeneralGateTest, InstrumentGate) {
	auto gate1 = gate::merge(gate::P0(0), gate::P0(1));
	auto gate2 = gate::merge(gate::P0(0), gate::P1(1));
	auto gate3 = gate::merge(gate::P1(0), gate::P0(1));
	auto gate4 = gate::merge(gate::P1(0), gate::P1(1));

	auto Inst = gate::Instrument({ gate3, gate2, gate1, gate4 }, 1);
	DensityMatrix s(3);
	s.set_computational_basis(0);
	Inst->update_quantum_state(&s);
	UINT res1 = s.get_classical_value(1);
	ASSERT_EQ(res1, 2);
	s.set_Haar_random_state();
	Inst->update_quantum_state(&s);
	UINT res2 = s.get_classical_value(1);
	delete Inst;
}

TEST(DensityMatrixGeneralGateTest, AdaptiveGate) {
	auto x = gate::X(0);
	auto adaptive = gate::Adaptive(x, [](const std::vector<UINT>& vec) { return vec[2] == 1; });
	DensityMatrix s(1);
	s.set_computational_basis(0);
	s.set_classical_value(2, 1);
	adaptive->update_quantum_state(&s);
	s.set_classical_value(2, 0);
	adaptive->update_quantum_state(&s);
	delete adaptive;
}


TEST(DensityMatrixGeneralGateTest, CheckProbabilisticGate) {
	const UINT n = 5;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;
	const UINT gate_count = 5;

	Random random;
	DensityMatrix state(n);

	std::vector<UINT> arr(n);
	std::iota(arr.begin(), arr.end(), 0);

	for (UINT repeat = 0; repeat < 10; ++repeat) {
		// create dist
		std::vector<double> probs;
		for (UINT i = 0; i < gate_count; ++i) probs.push_back(random.uniform());
		double sum = std::accumulate(probs.begin(), probs.end(), 0.);
		for (UINT i = 0; i < gate_count; ++i) probs[i] /= sum;

		// create gate list
		std::vector<QuantumGateBase*> gate_list;
		for (UINT i = 0; i < gate_count; ++i) {
			auto gate = gate::RandomUnitary(arr);
			gate_list.push_back(gate);
		}
		auto prob_gate = gate::Probabilistic(probs, gate_list);

		// update density matrix
		DensityMatrix dm(n);
		dm.set_Haar_random_state();

		// update by matrix reps
		ComplexMatrix mat = ComplexMatrix::Zero(dim, dim);
		for (UINT i = 0; i < gate_count; ++i) {
			ComplexMatrix gate_mat;
			gate_list[i]->set_matrix(gate_mat);
			ComplexMatrix dense_mat(dim, dim);
			for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) dense_mat(i, j) = dm.data_cpp()[i*dim + j];
			mat += probs[i] * gate_mat * dense_mat * gate_mat.adjoint();
		}
		prob_gate->update_quantum_state(&dm);

		// check equivalence
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(dm.data_cpp()[i*dim + j] - mat(i, j)), 0., eps);
		// check TP
		ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

		// release
		delete prob_gate;
		for (UINT i = 0; i < gate_count; ++i) {
			delete gate_list[i];
		}
	}
}



TEST(DensityMatrixGeneralGateTest, CheckCPTPMap) {
	const UINT n = 2;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;
	const UINT gate_count = 5;

	Random random;
	DensityMatrix state(n);

	std::vector<UINT> arr(n);
	std::iota(arr.begin(), arr.end(), 0);

	for (UINT repeat = 0; repeat < 10; ++repeat) {
		// create dist
		std::vector<double> probs;
		for (UINT i = 0; i < gate_count; ++i) probs.push_back(random.uniform());
		double sum = std::accumulate(probs.begin(), probs.end(), 0.);
		for (UINT i = 0; i < gate_count; ++i) probs[i] /= sum;

		// create not TP gate list
		std::vector<QuantumGateBase*> gate_list;
		for (UINT i = 0; i < gate_count; ++i) {
			auto gate = gate::RandomUnitary(arr);
			gate->multiply_scalar(sqrt(probs[i]));
			gate_list.push_back(gate);
		}
		auto cptp_gate = gate::CPTP(gate_list);

		// update density matrix
		DensityMatrix dm(n);
		dm.set_Haar_random_state();

		// update by matrix reps
		ComplexMatrix mat = ComplexMatrix::Zero(dim, dim);
		for (UINT i = 0; i < gate_count; ++i) {
			ComplexMatrix gate_mat;
			gate_list[i]->set_matrix(gate_mat);
			ComplexMatrix dense_mat(dim, dim);
			for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) dense_mat(i, j) = dm.data_cpp()[i*dim + j];
			mat += gate_mat * dense_mat * gate_mat.adjoint();
		}
		cptp_gate->update_quantum_state(&dm);

		// check equivalence
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(dm.data_cpp()[i*dim + j] - mat(i, j)), 0., eps);
		// check TP
		ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

		// release
		delete cptp_gate;
		for (UINT i = 0; i < gate_count; ++i) {
			delete gate_list[i];
		}
	}
}

TEST(DensityMatrixGeneralGateTest, AmplitudeDampingTest) {
	const UINT n = 1;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;

	Random random;
	DensityMatrix state(n);

	std::vector<UINT> arr(n);
	std::iota(arr.begin(), arr.end(), 0);

	for (UINT repeat = 0; repeat < 10; ++repeat) {
		double prob = random.uniform();

		ComplexMatrix K0(2, 2), K1(2, 2);
		K0 << 1, 0, 0, sqrt(1 - prob);
		K1 << 0, sqrt(prob), 0, 0;

		auto gate0 = gate::DenseMatrix(arr, K0);
		auto gate1 = gate::DenseMatrix(arr, K1);
		std::vector<QuantumGateBase*> gate_list = { gate0, gate1 };
		auto cptp_gate = gate::CPTP(gate_list);

		// update density matrix
		DensityMatrix dm(n);
		dm.set_Haar_random_state();

		// update by matrix reps
		ComplexMatrix mat = ComplexMatrix::Zero(dim, dim);
		for (UINT i = 0; i < gate_list.size(); ++i) {
			ComplexMatrix gate_mat;
			gate_list[i]->set_matrix(gate_mat);
			ComplexMatrix dense_mat(dim, dim);
			for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) dense_mat(i, j) = dm.data_cpp()[i*dim + j];
			mat += gate_mat * dense_mat * gate_mat.adjoint();
		}
		cptp_gate->update_quantum_state(&dm);

		// check equivalence
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(dm.data_cpp()[i*dim + j] - mat(i, j)), 0., eps);
		// check TP
		ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

		// release
		delete cptp_gate;
		for (UINT i = 0; i < gate_list.size(); ++i) {
			delete gate_list[i];
		}
	}
}

TEST(DensityMatrixGeneralGateTest, DepolarizingTest) {
	const UINT n = 1;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;
	double prob = 0.2;

	Random random;
	DensityMatrix state(n);

	// update density matrix
	DensityMatrix dm(n);
	dm.set_Haar_random_state();
	ComplexMatrix dense_mat(dim, dim);
	for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) dense_mat(i, j) = dm.data_cpp()[i*dim + j];
	ASSERT_NEAR(dense_mat.norm(), 1., eps);
	ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

	auto conv_mat = dense_mat * (1 - prob) + prob / dim * ComplexMatrix::Identity(dim, dim);
	auto two_qubit_depolarizing = gate::DepolarizingNoise(0,prob*3/4);
	two_qubit_depolarizing->update_quantum_state(&dm);
	ASSERT_NEAR(dense_mat.norm(), 1., eps);
	ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

	//std::cout << dense_mat << std::endl;
	//std::cout << dm << std::endl;

	// check equivalence
	for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(dm.data_cpp()[i*dim + j] - conv_mat(i, j)), 0., eps);
	delete two_qubit_depolarizing;
}


TEST(DensityMatrixGeneralGateTest, TwoQubitDepolarizingTest) {
	const UINT n = 2;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;
	double prob = 0.2;

	Random random;
	DensityMatrix state(n);

	// update density matrix
	DensityMatrix dm(n);
	dm.set_Haar_random_state();
	ComplexMatrix dense_mat(dim, dim);
	for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) dense_mat(i, j) = dm.data_cpp()[i*dim + j];
	ASSERT_NEAR(dense_mat.norm(), 1., eps);
	ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

	//std::cout << dense_mat << std::endl;
	//std::cout << dm << std::endl;
	auto conv_mat = dense_mat * (1 - prob) + prob / dim * ComplexMatrix::Identity(dim, dim);
	auto two_qubit_depolarizing = gate::TwoQubitDepolarizingNoise(0, 1, prob * 15 / 16);
	two_qubit_depolarizing->update_quantum_state(&dm);
	//std::cout << conv_mat << std::endl;
	//std::cout << dm << std::endl;
	ASSERT_NEAR(dense_mat.norm(), 1., eps);
	ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

	// check equivalence
	for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(dm.data_cpp()[i*dim + j] - conv_mat(i, j)), 0., eps);
	// check TP
	delete two_qubit_depolarizing;
}


/*
// not implemented yet
TEST(DensityMatrixGateTest, ReversibleBooleanGate) {
	const double eps = 1e-14;
	std::function<ITYPE(ITYPE, ITYPE)> func = [](ITYPE index, ITYPE dim) -> ITYPE {
		return (index + 1) % dim;
	};
	std::vector<UINT> target_qubit = { 2,0 };
	auto gate = gate::ReversibleBoolean(target_qubit, func);
	ComplexMatrix cm;
	gate->set_matrix(cm);
	QuantumState state(3);
	gate->update_quantum_state(&state);
	ASSERT_NEAR(abs(state.data_cpp()[4] - 1.), 0, eps);
	gate->update_quantum_state(&state);
	ASSERT_NEAR(abs(state.data_cpp()[1] - 1.), 0, eps);
	gate->update_quantum_state(&state);
	ASSERT_NEAR(abs(state.data_cpp()[5] - 1.), 0, eps);
	gate->update_quantum_state(&state);
	ASSERT_NEAR(abs(state.data_cpp()[0] - 1.), 0, eps);
}

*/