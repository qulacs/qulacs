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



TEST(DensityMatrixGateTest, ApplySingleQubitGate) {

	Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2), H(2, 2), S(2, 2), T(2, 2), sqrtX(2, 2), sqrtY(2, 2), P0(2, 2), P1(2, 2);

	Identity << 1, 0, 0, 1;
	X << 0, 1, 1, 0;
	Y << 0, -1.i, 1.i, 0;
	Z << 1, 0, 0, -1;
	H << 1, 1, 1, -1; H /= sqrt(2.);
	S << 1, 0, 0, 1.i;
	T << 1, 0, 0, (1. + 1.i) / sqrt(2.);
	sqrtX << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
	sqrtY << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
	P0 << 1, 0, 0, 0;
	P1 << 0, 0, 0, 1;


	const UINT n = 5;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;

	Random random;
	DensityMatrix state(n);
	std::vector< std::pair< std::function<QuantumGateBase*(UINT)>, Eigen::MatrixXcd >> funclist;
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

	QuantumState test_state(n);
	for (UINT repeat = 0; repeat < 10; ++repeat) {
		for (auto func_mat : funclist) {
			auto func = func_mat.first;
			auto mat = func_mat.second;
			UINT target = random.int32() % n;

			test_state.set_Haar_random_state();
			state.load(&test_state);

			auto gate = func(target);
			gate->update_quantum_state(&state);
			gate->update_quantum_state(&test_state);
			ComplexMatrix small_mat;
			gate->set_matrix(small_mat);

			DensityMatrix dm_test(n);
			dm_test.load(&test_state);
			for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm_test.data_cpp()[i*dim + j]), 0, eps);
		}
	}
}


TEST(DensityMatrixGateTest, ApplySingleQubitRotationGate) {

	Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);

	Identity << 1, 0, 0, 1;
	X << 0, 1, 1, 0;
	Y << 0, -1.i, 1.i, 0;
	Z << 1, 0, 0, -1;

	const UINT n = 5;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;

	Random random;
	DensityMatrix state(n);
	QuantumState test_state(n);
	std::vector< std::pair< std::function<QuantumGateBase*(UINT, double)>, Eigen::MatrixXcd >> funclist;
	funclist.push_back(std::make_pair(gate::RX, X));
	funclist.push_back(std::make_pair(gate::RY, Y));
	funclist.push_back(std::make_pair(gate::RZ, Z));

	for (UINT repeat = 0; repeat < 10; ++repeat) {
		for (auto func_mat : funclist) {
			UINT target = random.int32() % n;
			double angle = random.uniform() * 3.14159;

			auto func = func_mat.first;
			auto mat = cos(angle / 2) * Eigen::MatrixXcd::Identity(2, 2) + 1.i * sin(angle / 2)* func_mat.second;

			test_state.set_Haar_random_state();
			state.load(&test_state);

			auto gate = func(target, angle);
			gate->update_quantum_state(&state);
			gate->update_quantum_state(&test_state);

			DensityMatrix dm_test(n);
			dm_test.load(&test_state);
			for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm_test.data_cpp()[i*dim + j]), 0, eps);
		}
	}
}

TEST(DensityMatrixGateTest, ApplyTwoQubitGate) {

	const UINT n = 5;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;

	Random random;
	DensityMatrix state(n);
	QuantumState test_state(n);
	std::vector< std::pair< std::function<QuantumGateBase*(UINT, UINT)>, std::function<Eigen::MatrixXcd(UINT, UINT, UINT)>>> funclist;
	funclist.push_back(std::make_pair(gate::CNOT, get_eigen_matrix_full_qubit_CNOT));
	funclist.push_back(std::make_pair(gate::CZ, get_eigen_matrix_full_qubit_CZ));

	for (UINT repeat = 0; repeat < 10; ++repeat) {
		for (auto func_mat : funclist) {
			UINT control = random.int32() % n;
			UINT target = random.int32() % n;
			if (target == control) target = (target + 1) % n;

			auto func = func_mat.first;
			auto func_eig = func_mat.second;

			test_state.set_Haar_random_state();
			state.load(&test_state);

			// update state
			auto gate = func(control, target);
			gate->update_quantum_state(&state);
			gate->update_quantum_state(&test_state);

			DensityMatrix dm_test(n);
			dm_test.load(&test_state);
			for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm_test.data_cpp()[i*dim + j]), 0, eps);
		}
	}

	funclist.clear();
	funclist.push_back(std::make_pair(gate::SWAP, get_eigen_matrix_full_qubit_SWAP));
	for (UINT repeat = 0; repeat < 10; ++repeat) {
		for (auto func_mat : funclist) {
			UINT control = random.int32() % n;
			UINT target = random.int32() % n;
			if (target == control) target = (target + 1) % n;

			auto func = func_mat.first;
			auto func_eig = func_mat.second;

			test_state.set_Haar_random_state();
			state.load(&test_state);

			auto gate = func(control, target);
			gate->update_quantum_state(&state);
			gate->update_quantum_state(&test_state);

			DensityMatrix dm_test(n);
			dm_test.load(&test_state);
			for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm_test.data_cpp()[i*dim + j]), 0, eps);
		}
	}
}


TEST(DensityMatrixGateTest, ApplyMultiQubitGate) {

	const UINT n = 4;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;

	Random random;
	DensityMatrix state(n);
	QuantumState test_state(n);
	std::vector< std::pair< std::function<QuantumGateBase*(UINT, UINT)>, std::function<Eigen::MatrixXcd(UINT, UINT, UINT)>>> funclist;

	//gate::DenseMatrix
	//gate::Pauli
	//gate::PauliRotation

	for (UINT repeat = 0; repeat < 10; ++repeat) {

		test_state.set_Haar_random_state();
		state.load(&test_state);

		PauliOperator pauli(1.0);
		for (UINT i = 0; i < n; ++i) {
			pauli.add_single_Pauli(i, random.int32() % 4);
		}
		auto gate = gate::Pauli(pauli.get_index_list(), pauli.get_pauli_id_list());
		gate->update_quantum_state(&state);
		gate->update_quantum_state(&test_state);

		DensityMatrix dm_test(n);
		dm_test.load(&test_state);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm_test.data_cpp()[i*dim + j]), 0, eps);
	}

	for (UINT repeat = 0; repeat < 10; ++repeat) {

		test_state.set_Haar_random_state();
		state.load(&test_state);

		PauliOperator pauli(1.0);
		for (UINT i = 0; i < n; ++i) {
			pauli.add_single_Pauli(i, random.int32() % 4);
		}
		double angle = random.uniform()*3.14159;

		auto gate = gate::PauliRotation(pauli.get_index_list(), pauli.get_pauli_id_list(), angle);

		gate->update_quantum_state(&state);
		gate->update_quantum_state(&test_state);

		DensityMatrix dm_test(n);
		dm_test.load(&test_state);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm_test.data_cpp()[i*dim + j]), 0, eps);
	}

}


TEST(DensityMatrixGateTest, MergeTensorProduct) {
	UINT n = 2;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;

	auto x0 = gate::X(0);
	auto y1 = gate::Y(1);
	auto xy01 = gate::merge(x0, y1);

	DensityMatrix state(n);
	QuantumState test_state(n);
	test_state.set_Haar_random_state();
	state.load(&test_state);

	xy01->update_quantum_state(&state);
	xy01->update_quantum_state(&test_state);

	DensityMatrix dm(n);
	dm.load(&test_state);
	for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim +j] - dm.data_cpp()[i*dim+j]), 0, eps);

	delete x0;
	delete y1;
	delete xy01;
}

TEST(DensityMatrixGateTest, MergeMultiply) {
	UINT n = 1;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;
	auto x0 = gate::X(0);
	auto y0 = gate::Y(0);

	//  U_{z0} = YX = -iZ
	auto xy00 = gate::merge(x0, y0);

	DensityMatrix state(n);
	QuantumState test_state(n);
	Eigen::VectorXcd test_state_eigen(dim);
	test_state.set_Haar_random_state();
	state.load(&test_state);

	xy00->update_quantum_state(&state);
	xy00->update_quantum_state(&test_state);

	DensityMatrix dm(n);
	dm.load(&test_state);
	for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm.data_cpp()[i*dim + j]), 0, eps);

	delete x0;
	delete y0;
	delete xy00;
}

TEST(DensityMatrixGateTest, MergeTensorProductAndMultiply) {
	UINT n = 2;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;

	auto x0 = gate::X(0);
	auto y1 = gate::Y(1);
	auto xy01 = gate::merge(x0, y1);
	auto iy01 = gate::merge(xy01, x0);

	// Expected : x_0 y_1 x_0 = y_1

	DensityMatrix state(n);
	QuantumState test_state(n);
	test_state.set_Haar_random_state();
	state.load(&test_state);

	iy01->update_quantum_state(&state);
	iy01->update_quantum_state(&test_state);

	DensityMatrix dm(n);
	dm.load(&test_state);
	for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm.data_cpp()[i*dim + j]), 0, eps);

	delete x0;
	delete y1;
	delete xy01;
	delete iy01;
}

TEST(DensityMatrixGateTest, RandomPauliMerge) {
	UINT n = 5;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;

	UINT gate_count = 10;
	UINT max_repeat = 3;
	Random random;
	random.set_seed(2);

	std::vector<UINT> new_pauli_ids = { 0,0,0,1 };
	std::vector<UINT> targets = { 0,1,2,2 };

	// define states
	DensityMatrix state(n);
	QuantumState test_state(n);

	for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
		// pick random state and copy to test
		test_state.set_Haar_random_state();
		state.load(&test_state);

		auto merged_gate = gate::Identity(0);
		QuantumGateMatrix* next_merged_gate = NULL;
		QuantumGateBase* new_gate = NULL;
		for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {

			// pick random pauli
			UINT new_pauli_id = random.int32() % 4;
			UINT target = random.int32() % n;
			if (new_pauli_id == 0) new_gate = gate::Identity(target);
			else if (new_pauli_id == 1) new_gate = gate::X(target);
			else if (new_pauli_id == 2) new_gate = gate::Y(target);
			else if (new_pauli_id == 3) new_gate = gate::Z(target);
			else FAIL();

			// create new gate with merge
			next_merged_gate = gate::merge(merged_gate, new_gate);
			delete merged_gate;
			merged_gate = next_merged_gate;
			next_merged_gate = NULL;

			delete new_gate;
		}
		merged_gate->update_quantum_state(&state);
		merged_gate->update_quantum_state(&test_state);

		DensityMatrix dm(n);
		dm.load(&test_state);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm.data_cpp()[i*dim + j]), 0, eps);

		delete merged_gate;

	}
}

TEST(DensityMatrixGateTest, RandomPauliRotationMerge) {
	UINT n = 5;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;

	UINT gate_count = 10;
	UINT max_repeat = 3;
	Random random;
	random.set_seed(2);

	std::vector<UINT> new_pauli_ids = { 0,0,0,1 };
	std::vector<UINT> targets = { 0,1,2,2 };

	// define states
	DensityMatrix state(n);
	QuantumState test_state(n);

	for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
		// pick random state and copy to test
		test_state.set_Haar_random_state();
		state.load(&test_state);

		auto merged_gate = gate::Identity(0);
		QuantumGateMatrix* next_merged_gate = NULL;
		QuantumGateBase* new_gate = NULL;

		for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {

			// pick random pauli
			UINT new_pauli_id = (random.int32() % 3) + 1;
			UINT target = random.int32() % n;
			double angle = random.uniform() * 3.14159;
			if (new_pauli_id == 1) new_gate = gate::RX(target, angle);
			else if (new_pauli_id == 2) new_gate = gate::RY(target, angle);
			else if (new_pauli_id == 3) new_gate = gate::RZ(target, angle);
			else FAIL();

			// create new gate with merge
			next_merged_gate = gate::merge(merged_gate, new_gate);
			delete merged_gate;
			merged_gate = next_merged_gate;
			next_merged_gate = NULL;
			delete new_gate;
		}
		merged_gate->update_quantum_state(&state);
		merged_gate->update_quantum_state(&test_state);
		delete merged_gate;

		DensityMatrix dm(n);
		dm.load(&test_state);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm.data_cpp()[i*dim + j]), 0, eps);
	}
}

TEST(DensityMatrixGateTest, RandomUnitaryMerge) {
	UINT n = 5;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;

	UINT gate_count = 10;
	UINT max_repeat = 3;
	Random random;
	random.set_seed(2);

	std::vector<UINT> new_pauli_ids = { 0,0,0,1 };
	std::vector<UINT> targets = { 0,1,2,2 };

	// define states
	DensityMatrix state(n);
	QuantumState test_state(n);

	for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
		// pick random state and copy to test
		test_state.set_Haar_random_state();
		state.load(&test_state);

		auto merged_gate = gate::Identity(0);
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
			di /= norm; dx /= norm; dy /= norm; dz /= norm;
			ComplexMatrix mat = di * get_eigen_matrix_single_Pauli(0) + 1.i*(dx*get_eigen_matrix_single_Pauli(1) + dy * get_eigen_matrix_single_Pauli(2) + dz * get_eigen_matrix_single_Pauli(3));

			auto new_gate = gate::DenseMatrix(target, mat);

			// create new gate with merge
			next_merged_gate = gate::merge(merged_gate, new_gate);
			delete merged_gate;
			merged_gate = next_merged_gate;
			next_merged_gate = NULL;

			// dispose picked pauli
			delete new_gate;
		}
		merged_gate->update_quantum_state(&state);
		merged_gate->update_quantum_state(&test_state);
		delete merged_gate;

		DensityMatrix dm(n);
		dm.load(&test_state);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm.data_cpp()[i*dim + j]), 0, eps);
	}
}

TEST(DensityMatrixGateTest, RandomUnitaryMergeLarge) {
	UINT n = 5;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;

	UINT gate_count = 5;
	UINT max_repeat = 2;
	Random random;
	random.set_seed(2);

	std::vector<UINT> new_pauli_ids = { 0,0,0,1 };
	std::vector<UINT> targets = { 0,1,2,2 };

	// define states
	DensityMatrix state(n),state2(n);
	QuantumState test_state(n);
	
	for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
		// pick random state and copy to test
		test_state.set_Haar_random_state();
		state.load(&test_state);
		state2.load(&test_state);

		auto merged_gate1 = gate::Identity(0);
		auto merged_gate2 = gate::Identity(0);
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
			di /= norm; dx /= norm; dy /= norm; dz /= norm;
			ComplexMatrix mat = di * get_eigen_matrix_single_Pauli(0) + 1.i*(dx*get_eigen_matrix_single_Pauli(1) + dy * get_eigen_matrix_single_Pauli(2) + dz * get_eigen_matrix_single_Pauli(3));

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
			di /= norm; dx /= norm; dy /= norm; dz /= norm;
			ComplexMatrix mat = di * get_eigen_matrix_single_Pauli(0) + 1.i*(dx*get_eigen_matrix_single_Pauli(1) + dy * get_eigen_matrix_single_Pauli(2) + dz * get_eigen_matrix_single_Pauli(3));

			auto new_gate = gate::DenseMatrix(target, mat);

			// create new gate with merge
			next_merged_gate = gate::merge(merged_gate2, new_gate);
			delete merged_gate2;
			merged_gate2 = next_merged_gate;
			next_merged_gate = NULL;

			// dispose picked pauli
			delete new_gate;
		}
		auto merged_gate = gate::merge(merged_gate1, merged_gate2);
		merged_gate->update_quantum_state(&state);
		merged_gate->update_quantum_state(&test_state);
		merged_gate1->update_quantum_state(&state2);
		merged_gate2->update_quantum_state(&state2);

		delete merged_gate;
		delete merged_gate1;
		delete merged_gate2;

		// check equivalence
		DensityMatrix dm(n);
		dm.load(&test_state);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm.data_cpp()[i*dim + j]), 0, eps);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - state2.data_cpp()[i*dim + j]), 0, eps);
	}
}


TEST(DensityMatrixGateTest, RandomControlMergeSmall) {
	UINT n = 4;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;

	UINT gate_count = 10;
	Random random;

	std::vector<UINT> arr;
	for (UINT i = 0; i < n; ++i) arr.push_back(i);

	for (gate_count = 1; gate_count < n * 2; ++gate_count) {
		ComplexMatrix mat = ComplexMatrix::Identity(dim, dim);
		DensityMatrix state(n);
		QuantumState test_state(n);

		test_state.set_Haar_random_state();
		state.load(&test_state);

		auto merge_gate1 = gate::Identity(0);
		auto merge_gate2 = gate::Identity(0);

		for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
			std::random_shuffle(arr.begin(), arr.end());
			UINT target = arr[0];
			UINT control = arr[1];
			auto new_gate = gate::CNOT(control, target);
			merge_gate1 = gate::merge(merge_gate1, new_gate);

			auto cmat = get_eigen_matrix_full_qubit_CNOT(control, target, n);
			mat = cmat * mat;
		}
		merge_gate1->update_quantum_state(&state);
		merge_gate1->update_quantum_state(&test_state);

		DensityMatrix dm(n);
		dm.load(&test_state);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm.data_cpp()[i*dim + j]), 0, eps);
	}
}


TEST(DensityMatrixGateTest, RandomControlMergeLarge) {
	UINT n = 4;
	ITYPE dim = 1ULL << n;
	const double eps = 1e-14;

	UINT gate_count = 10;
	Random random;

	std::vector<UINT> arr;
	for (UINT i = 0; i < n; ++i) arr.push_back(i);

	for (gate_count = 1; gate_count < n * 2; ++gate_count) {
		ComplexMatrix mat = ComplexMatrix::Identity(dim, dim);
		DensityMatrix state(n), state2(n);
		QuantumState test_state(n);

		test_state.set_Haar_random_state();
		state.load(&test_state);
		state2.load(&test_state);

		auto merge_gate1 = gate::Identity(0);
		auto merge_gate2 = gate::Identity(0);

		for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
			std::random_shuffle(arr.begin(), arr.end());
			UINT target = arr[0];
			UINT control = arr[1];
			auto new_gate = gate::CNOT(control, target);
			merge_gate1 = gate::merge(merge_gate1, new_gate);

			auto cmat = get_eigen_matrix_full_qubit_CNOT(control, target, n);
			mat = cmat * mat;
		}

		for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
			std::random_shuffle(arr.begin(), arr.end());
			UINT target = arr[0];
			UINT control = arr[1];
			auto new_gate = gate::CNOT(control, target);
			merge_gate2 = gate::merge(merge_gate2, new_gate);

			auto cmat = get_eigen_matrix_full_qubit_CNOT(control, target, n);
			mat = cmat * mat;
		}

		auto merge_gate = gate::merge(merge_gate1, merge_gate2);
		merge_gate->update_quantum_state(&state);
		merge_gate->update_quantum_state(&test_state);
		merge_gate1->update_quantum_state(&state2);
		merge_gate2->update_quantum_state(&state2);

		DensityMatrix dm(n);
		dm.load(&test_state);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm.data_cpp()[i*dim + j]), 0, eps);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - state2.data_cpp()[i*dim + j]), 0, eps);
	}
}







TEST(DensityMatrixGateTest, MultiTarget) {
	const UINT n = 8;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;
	const UINT target_count = 5;

	Random random;
	DensityMatrix state(n);
	QuantumState test_state(n);
	std::vector<UINT> arr;
	for (UINT i = 0; i < n; ++i) arr.push_back(i);

	for (UINT repeat = 0; repeat < 10; ++repeat) {
		std::random_shuffle(arr.begin(), arr.end());
		std::vector<UINT> target_list;
		for (UINT i = 0; i < target_count; ++i) target_list.push_back(arr[i]);
		auto gate = gate::RandomUnitary(target_list);

		test_state.set_Haar_random_state();
		state.load(&test_state);

		// update state
		gate->update_quantum_state(&state);
		gate->update_quantum_state(&test_state);
		delete gate;

		DensityMatrix dm_test(n);
		dm_test.load(&test_state);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm_test.data_cpp()[i*dim + j]), 0, eps);
	}
}


TEST(DensityMatrixGateTest, MultiControlSingleTarget) {
	const UINT n = 8;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;
	const UINT control_count = 3;

	Random random;
	DensityMatrix state(n);
	QuantumState test_state(n);
	std::vector<UINT> arr;
	for (UINT i = 0; i < n; ++i) arr.push_back(i);

	for (UINT repeat = 0; repeat < 10; ++repeat) {
		std::random_shuffle(arr.begin(), arr.end());
		std::vector<UINT> target_list;
		target_list.push_back(arr[0]);
		auto gate = gate::RandomUnitary(target_list);
		for (UINT i = 0; i < control_count; ++i) {
			gate->add_control_qubit(arr[i + 1], random.int32() % 2);
		}

		test_state.set_Haar_random_state();
		state.load(&test_state);

		// update state
		gate->update_quantum_state(&state);
		gate->update_quantum_state(&test_state);
		delete gate;

		DensityMatrix dm_test(n);
		dm_test.load(&test_state);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm_test.data_cpp()[i*dim + j]), 0, eps);
	}
}


TEST(DensityMatrixGateTest, SingleControlMultiTarget) {
	const UINT n = 8;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;
	const UINT target_count = 3;

	Random random;
	DensityMatrix state(n);
	QuantumState test_state(n);
	std::vector<UINT> arr;
	for (UINT i = 0; i < n; ++i) arr.push_back(i);

	for (UINT repeat = 0; repeat < 10; ++repeat) {
		std::random_shuffle(arr.begin(), arr.end());
		std::vector<UINT> target_list;
		for (UINT i = 0; i < target_count; ++i) target_list.push_back(arr[i]);
		UINT control = arr[target_count+1];
		UINT control_value = random.int32() % 2;
		auto gate = gate::RandomUnitary(target_list);
		gate->add_control_qubit(control, control_value);

		test_state.set_Haar_random_state();
		state.load(&test_state);

		// update state
		gate->update_quantum_state(&state);
		gate->update_quantum_state(&test_state);
		delete gate;

		DensityMatrix dm_test(n);
		dm_test.load(&test_state);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm_test.data_cpp()[i*dim + j]), 0, eps);
	}
}

TEST(DensityMatrixGateTest, MultiControlMultiTarget) {
	const UINT n = 8;
	const ITYPE dim = 1ULL << n;
	double eps = 1e-15;
	const UINT control_count = 3;
	const UINT target_count = 3;

	Random random;
	DensityMatrix state(n);
	QuantumState test_state(n);
	std::vector<UINT> arr;
	for (UINT i = 0; i < n; ++i) arr.push_back(i);

	for (UINT repeat = 0; repeat < 10; ++repeat) {
		std::random_shuffle(arr.begin(), arr.end());
		std::vector<UINT> target_list;
		for (UINT i = 0; i < target_count; ++i) target_list.push_back(arr[i]);
		auto gate = gate::RandomUnitary(target_list);
		for (UINT i = 0; i < control_count; ++i) {
			gate->add_control_qubit(arr[target_count + i], random.int32() % 2);
		}

		test_state.set_Haar_random_state();
		state.load(&test_state);

		// update state
		gate->update_quantum_state(&state);
		gate->update_quantum_state(&test_state);
		delete gate;

		DensityMatrix dm_test(n);
		dm_test.load(&test_state);
		for (ITYPE i = 0; i < dim; ++i) for (ITYPE j = 0; j < dim; ++j) ASSERT_NEAR(abs(state.data_cpp()[i*dim + j] - dm_test.data_cpp()[i*dim + j]), 0, eps);
	}
}

