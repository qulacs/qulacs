#include <gtest/gtest.h>
#include "../util/util.h"

#include <cmath>
#include <cppsim/state_gpu.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/pauli_operator.hpp>

#include <cppsim/utility.hpp>
#include <csim/update_ops.h>
#include <functional>



inline void set_eigen_from_gpu(ComplexVector& dst, QuantumStateGpu& src, ITYPE dim) {
	auto ptr = src.duplicate_data_cpp();
	for (ITYPE i = 0; i < dim; ++i) dst[i] = ptr[i];
	free(ptr);
}

inline void assert_eigen_eq_gpu(ComplexVector& v1, QuantumStateGpu& v2, ITYPE dim, double eps) {
	auto ptr = v2.duplicate_data_cpp();
	for (UINT i = 0; i < dim; ++i) {
		ASSERT_NEAR(ptr[i].real(), v1[i].real(), eps);
		ASSERT_NEAR(ptr[i].imag(), v1[i].imag(), eps);
	}
	free(ptr);
}

inline void assert_cpu_eq_gpu(QuantumStateCpu& state_cpu, QuantumStateGpu& state_gpu, ITYPE dim, double eps) {
	auto gpu_state_vector = state_gpu.duplicate_data_cpp();
	for (ITYPE i = 0; i < dim; ++i) {
		ASSERT_NEAR(state_cpu.data_cpp()[i].real(), gpu_state_vector[i].real(), eps);
		ASSERT_NEAR(state_cpu.data_cpp()[i].imag(), gpu_state_vector[i].imag(), eps);
	}
	free(gpu_state_vector);
}

inline void assert_gpu_eq_gpu(QuantumStateGpu& state_gpu1, QuantumStateGpu& state_gpu2, ITYPE dim, double eps) {
	auto gpu_state_vector1 = state_gpu1.duplicate_data_cpp();
	auto gpu_state_vector2 = state_gpu2.duplicate_data_cpp();
	for (ITYPE i = 0; i < dim; ++i) {
		ASSERT_NEAR(gpu_state_vector1[i].real(), gpu_state_vector2[i].real(), eps);
		ASSERT_NEAR(gpu_state_vector1[i].imag(), gpu_state_vector2[i].imag(), eps);
	}
	free(gpu_state_vector1);
	free(gpu_state_vector2);
}


TEST(GateTest, ApplySingleQubitGate) {

    Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2), H(2, 2), S(2, 2), T(2, 2), sqrtX(2, 2), sqrtY(2, 2), P0(2, 2), P1(2, 2);

    Identity << 1, 0, 0, 1;
    X << 0, 1, 1, 0;
    Y << 0, -1.i, 1.i, 0;
    Z << 1, 0, 0, -1;
    H << 1, 1, 1, -1; H /= sqrt(2.);
    S << 1, 0, 0, 1.i;
    T << 1, 0, 0, (1. + 1.i) /sqrt(2.);
    sqrtX << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
    sqrtY << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;


    const UINT n = 5;
    const ITYPE dim = 1ULL << n;
    double eps = 1e-15;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		Random random;
		QuantumStateGpu state(n, idx);
		std::vector< std::pair< std::function<QuantumGateBase* (UINT)>, Eigen::MatrixXcd >> funclist;
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

		Eigen::VectorXcd test_state1 = Eigen::VectorXcd::Zero(dim);
		Eigen::VectorXcd test_state2 = Eigen::VectorXcd::Zero(dim);
		for (UINT repeat = 0; repeat < 10; ++repeat) {
			for (auto func_mat : funclist) {
				auto func = func_mat.first;
				auto mat = func_mat.second;
				UINT target = random.int32() % n;

				state.set_Haar_random_state();
				set_eigen_from_gpu(test_state1, state, dim);
				set_eigen_from_gpu(test_state2, state, dim);

				auto gate = func(target);
				gate->update_quantum_state(&state);
				ComplexMatrix small_mat;
				gate->set_matrix(small_mat);
				test_state1 = get_expanded_eigen_matrix_with_identity(target, small_mat, n) * test_state1;
				test_state2 = get_expanded_eigen_matrix_with_identity(target, mat, n) * test_state2;

				assert_eigen_eq_gpu(test_state1, state, dim, eps);
				assert_eigen_eq_gpu(test_state2, state, dim, eps);
			}
		}
	}
}



TEST(GateTest, ApplySingleQubitRotationGate) {

    Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);

    Identity << 1, 0, 0, 1;
    X << 0, 1, 1, 0;
    Y << 0, -1.i, 1.i, 0;
    Z << 1, 0, 0, -1;

    const UINT n = 5;
    const ITYPE dim = 1ULL << n;
    double eps = 1e-15;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		Random random;
		QuantumStateGpu state(n, idx);
		std::vector< std::pair< std::function<QuantumGateBase* (UINT, double)>, Eigen::MatrixXcd >> funclist;
		funclist.push_back(std::make_pair(gate::RX, X));
		funclist.push_back(std::make_pair(gate::RY, Y));
		funclist.push_back(std::make_pair(gate::RZ, Z));

		Eigen::VectorXcd test_state1 = Eigen::VectorXcd::Zero(dim);
		Eigen::VectorXcd test_state2 = Eigen::VectorXcd::Zero(dim);
		for (UINT repeat = 0; repeat < 10; ++repeat) {
			for (auto func_mat : funclist) {
				UINT target = random.int32() % n;
				double angle = random.uniform() * 3.14159;

				auto func = func_mat.first;
				auto mat = cos(angle / 2) * Eigen::MatrixXcd::Identity(2, 2) + 1.i * sin(angle / 2) * func_mat.second;

				state.set_Haar_random_state();
				set_eigen_from_gpu(test_state1, state, dim);
				set_eigen_from_gpu(test_state2, state, dim);

				auto gate = func(target, angle);
				gate->update_quantum_state(&state);
				ComplexMatrix small_mat;
				gate->set_matrix(small_mat);
				test_state1 = get_expanded_eigen_matrix_with_identity(target, small_mat, n) * test_state1;
				test_state2 = get_expanded_eigen_matrix_with_identity(target, mat, n) * test_state2;

				assert_eigen_eq_gpu(test_state1, state, dim, eps);
				assert_eigen_eq_gpu(test_state2, state, dim, eps);
			}
		}
	}
}


TEST(GateTest, ApplyTwoQubitGate) {

    const UINT n = 5;
    const ITYPE dim = 1ULL << n;
    double eps = 1e-15;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		Random random;
		QuantumStateGpu state(n, idx), test_state(n, idx);
		std::vector< std::pair< std::function<QuantumGateBase* (UINT, UINT)>, std::function<Eigen::MatrixXcd(UINT, UINT, UINT)>>> funclist;
		funclist.push_back(std::make_pair(gate::CNOT, get_eigen_matrix_full_qubit_CNOT));
		funclist.push_back(std::make_pair(gate::CZ, get_eigen_matrix_full_qubit_CZ));

		Eigen::VectorXcd test_state1 = Eigen::VectorXcd::Zero(dim);
		for (UINT repeat = 0; repeat < 10; ++repeat) {
			for (auto func_mat : funclist) {
				UINT control = random.int32() % n;
				UINT target = random.int32() % n;
				if (target == control) target = (target + 1) % n;

				auto func = func_mat.first;
				auto func_eig = func_mat.second;

				state.set_Haar_random_state();
				test_state.load(&state);
				set_eigen_from_gpu(test_state1, state, dim);

				// update state
				auto gate = func(control, target);
				gate->update_quantum_state(&state);

				// update eigen state
				Eigen::MatrixXcd large_mat = func_eig(control, target, n);
				test_state1 = large_mat * test_state1;

				// update dense state
				ComplexMatrix small_mat;
				gate->set_matrix(small_mat);
				auto gate_dense = new QuantumGateMatrix(gate->target_qubit_list, small_mat, gate->control_qubit_list);
				gate_dense->update_quantum_state(&test_state);
				delete gate_dense;

				assert_eigen_eq_gpu(test_state1, state, dim, eps);
				assert_gpu_eq_gpu(state, test_state, dim, eps);
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

				state.set_Haar_random_state();
				test_state.load(&state);
				set_eigen_from_gpu(test_state1, state, dim);

				auto gate = func(control, target);
				gate->update_quantum_state(&state);

				Eigen::MatrixXcd large_mat = func_eig(control, target, n);
				test_state1 = large_mat * test_state1;

				// update dense state
				ComplexMatrix small_mat;
				gate->set_matrix(small_mat);
				auto gate_dense = new QuantumGateMatrix(gate->target_qubit_list, small_mat, gate->control_qubit_list);
				gate_dense->update_quantum_state(&test_state);
				delete gate_dense;

				//assert_eigen_eq_gpu(test_state1, state, dim, eps);
				//assert_gpu_eq_gpu(state, test_state, dim, eps);
			}
		}
	}
}


TEST(GateTest, ApplyMultiQubitGate) {


    const UINT n = 1;
    const ITYPE dim = 1ULL << n;
    double eps = 1e-15;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		Random random;
		QuantumStateGpu state(n, idx);
		std::vector< std::pair< std::function<QuantumGateBase* (UINT, UINT)>, std::function<Eigen::MatrixXcd(UINT, UINT, UINT)>>> funclist;

		//gate::DenseMatrix
		//gate::Pauli
		//gate::PauliRotation

		Eigen::VectorXcd test_state1 = Eigen::VectorXcd::Zero(dim);
		for (UINT repeat = 0; repeat < 10; ++repeat) {

			state.set_Haar_random_state();
			state.set_computational_basis(0);
			set_eigen_from_gpu(test_state1, state, dim);

			PauliOperator pauli(1.0);
			for (UINT i = 0; i < n; ++i) {
				pauli.add_single_Pauli(i, random.int32() % 4);
			}
			auto gate = gate::Pauli(pauli.get_index_list(), pauli.get_pauli_id_list());
			Eigen::MatrixXcd large_mat = get_eigen_matrix_full_qubit_pauli(pauli.get_index_list(), pauli.get_pauli_id_list(), n);
			test_state1 = large_mat * test_state1;

			std::vector<UINT> target_list, control_list;
			ComplexMatrix small_mat;
			gate->set_matrix(small_mat);
			auto gate_dense = new QuantumGateMatrix(gate->target_qubit_list, small_mat, gate->control_qubit_list);
			gate_dense->update_quantum_state(&state);
			delete gate_dense;

			//std::cout << state << std::endl << test_state1 << std::endl;
			//std::cout << small_mat << std::endl << large_mat << std::endl;
			//for (UINT i = 0; i < 4; ++i) std::cout << small_mat.data()[i] << std::endl;

			assert_eigen_eq_gpu(test_state1, state, dim, eps);
		}

		for (UINT repeat = 0; repeat < 10; ++repeat) {

			state.set_Haar_random_state();
			set_eigen_from_gpu(test_state1, state, dim);

			PauliOperator pauli(1.0);
			for (UINT i = 0; i < n; ++i) {
				pauli.add_single_Pauli(i, random.int32() % 4);
			}
			double angle = random.uniform() * 3.14159;

			Eigen::MatrixXcd large_mat = cos(angle / 2) * Eigen::MatrixXcd::Identity(dim, dim) + 1.i * sin(angle / 2) * get_eigen_matrix_full_qubit_pauli(pauli.get_index_list(), pauli.get_pauli_id_list(), n);
			test_state1 = large_mat * test_state1;

			auto gate = gate::PauliRotation(pauli.get_index_list(), pauli.get_pauli_id_list(), angle);
			std::vector<UINT> target_list, control_list;
			ComplexMatrix small_mat;
			gate->set_matrix(small_mat);
			auto gate_dense = new QuantumGateMatrix(gate->target_qubit_list, small_mat, gate->control_qubit_list);
			gate_dense->update_quantum_state(&state);
			delete gate_dense;

			assert_eigen_eq_gpu(test_state1, state, dim, eps);
		}
	}
}


TEST(GateTest, MergeTensorProduct) {
    UINT n = 2;
    ITYPE dim = 1ULL << n;
    const double eps = 1e-14;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		auto x0 = gate::X(0);
		auto y1 = gate::Y(1);
		auto xy01 = gate::merge(x0, y1);

		QuantumStateGpu state(n, idx), test_state(n, idx);
		Eigen::VectorXcd test_state_eigen(dim);
		state.set_Haar_random_state();

		test_state.load(&state);
		set_eigen_from_gpu(test_state_eigen, state, dim);
		//assert_gpu_eq_gpu(state, test_state, dim, eps);

		xy01->update_quantum_state(&state);
		x0->update_quantum_state(&test_state);
		y1->update_quantum_state(&test_state);

		Eigen::MatrixXcd mat = get_eigen_matrix_full_qubit_pauli({ 1,2 });
		test_state_eigen = mat * test_state_eigen;

		assert_eigen_eq_gpu(test_state_eigen, state, dim, eps);
		assert_eigen_eq_gpu(test_state_eigen, test_state, dim, eps);
		assert_gpu_eq_gpu(state, test_state, dim, eps);
		
		delete x0;
		delete y1;
		delete xy01;
	}
}



TEST(GateTest, MergeMultiply) {
    UINT n = 1;
    ITYPE dim = 1ULL << n;
    const double eps = 1e-14;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		auto x0 = gate::X(0);
		auto y0 = gate::Y(0);
		//  U_{z0} = YX = -iZ
		auto xy00 = gate::merge(x0, y0);
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);

		QuantumStateGpu state(n, idx), test_state(n, idx);
		Eigen::VectorXcd test_state_eigen(dim);
		state.set_Haar_random_state();

		test_state.load(&state);
		set_eigen_from_gpu(test_state_eigen, state, dim);

		xy00->update_quantum_state(&state);
		x0->update_quantum_state(&test_state);
		y0->update_quantum_state(&test_state);
		Eigen::MatrixXcd mat = -1.i * get_eigen_matrix_full_qubit_pauli({ 3 });
		test_state_eigen = mat * test_state_eigen;

		assert_eigen_eq_gpu(test_state_eigen, state, dim, eps);
		assert_gpu_eq_gpu(state, test_state, dim, eps);
		delete x0;
		delete y0;
		delete xy00;
	}
}


TEST(GateTest, MergeTensorProductAndMultiply) {
    UINT n = 2;
    ITYPE dim = 1ULL << n;
    const double eps = 1e-14;

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		auto x0 = gate::X(0);
		auto y1 = gate::Y(1);
		auto xy01 = gate::merge(x0, y1);
		//std::cout << xy01 << std::endl;
		auto iy01 = gate::merge(xy01, x0);
		// Expected : x_0 y_1 x_0 = y_1

		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);

		QuantumStateGpu state(n, idx), test_state(n, idx);
		Eigen::VectorXcd test_state_eigen(dim);
		state.set_Haar_random_state();

		test_state.load(&state);
		set_eigen_from_gpu(test_state_eigen, state, dim);

		iy01->update_quantum_state(&state);
		y1->update_quantum_state(&test_state);
		Eigen::MatrixXcd mat = get_eigen_matrix_full_qubit_pauli({ 0,2 });
		test_state_eigen = mat * test_state_eigen;

		//std::cout << iy01 << std::endl << std::endl;
		//std::cout << mat << std::endl;

		assert_eigen_eq_gpu(test_state_eigen, state, dim, eps);
		assert_gpu_eq_gpu(state, test_state, dim, eps);
		delete x0;
		delete y1;
		delete xy01;
		delete iy01;
	}
}

TEST(GateTest, RandomPauliMerge) {
    UINT n = 5;
    ITYPE dim = 1ULL << n;
    const double eps = 1e-14;

    UINT gate_count = 10;
    UINT max_repeat = 3;
    Random random;
    random.set_seed(2);

    std::vector<UINT> new_pauli_ids = { 0,0,0,1 };
    std::vector<UINT> targets = { 0,1,2,2 };

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		// define states
		QuantumStateGpu state(n, idx), test_state(n, idx);
		Eigen::VectorXcd test_state_eigen(dim);

		for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
			// pick random state and copy to test
			state.set_Haar_random_state();
			test_state.load(&state);
			set_eigen_from_gpu(test_state_eigen, state, dim);

			// check equivalence
			assert_eigen_eq_gpu(test_state_eigen, state, dim, eps);
			assert_gpu_eq_gpu(test_state, state, dim, eps);

			auto merged_gate = gate::Identity(0);
			QuantumGateMatrix* next_merged_gate = NULL;
			QuantumGateBase* new_gate = NULL;
			Eigen::MatrixXcd total_matrix = Eigen::MatrixXcd::Identity(dim, dim);
			//std::cout << "initial state : " << state << std::endl;

			for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {

				// pick random pauli
				UINT new_pauli_id = random.int32() % 4;
				UINT target = random.int32() % n;
				//UINT new_pauli_id = new_pauli_ids[gate_index];
				//UINT target = targets[gate_index];
				if (new_pauli_id == 0) new_gate = gate::Identity(target);
				else if (new_pauli_id == 1) new_gate = gate::X(target);
				else if (new_pauli_id == 2) new_gate = gate::Y(target);
				else if (new_pauli_id == 3) new_gate = gate::Z(target);
				else FAIL();

				//std::cout << "***************************************************" << std::endl;
				//std::cout << " ***** Pauli = " << new_pauli_id << " at " << target << std::endl;

				// create new gate with merge
				next_merged_gate = gate::merge(merged_gate, new_gate);
				delete merged_gate;
				merged_gate = next_merged_gate;
				next_merged_gate = NULL;

				// update test state with latest gate
				new_gate->update_quantum_state(&test_state);

				// update eigen state with matrix mul
				auto new_gate_matrix = get_expanded_eigen_matrix_with_identity(target, get_eigen_matrix_single_Pauli(new_pauli_id), n);
				total_matrix = new_gate_matrix * total_matrix;
				test_state_eigen = new_gate_matrix * test_state_eigen;

				ComplexMatrix check_mat;
				merged_gate->set_matrix(check_mat);
				if (check_mat.rows() == total_matrix.rows()) {
					for (ITYPE x = 0; x < dim; ++x) {
						for (ITYPE y = 0; y < dim; ++y) {
							ASSERT_NEAR(abs(total_matrix(x, y) - check_mat(x, y)), 0, eps) << (QuantumGateMatrix*)merged_gate << std::endl << "current eigen matrix : \n" << total_matrix << std::endl;
						}
					}
				}

				//std::cout << "current state : " << test_state << std::endl << "current eigen state : \n" << test_state_eigen << std::endl;
				//std::cout << "initial matrix : " << (QuantumGateMatrix*)merged_gate << std::endl << "current eigen matrix : \n" << total_matrix << std::endl;
				//std::cout << "***************************************************" << std::endl;

				// dispose picked pauli
				delete new_gate;
			}
			merged_gate->update_quantum_state(&state);
			delete merged_gate;
			// check equivalence
			assert_eigen_eq_gpu(test_state_eigen, state, dim, eps);
			assert_gpu_eq_gpu(test_state, state, dim, eps);
		}
	}
}


TEST(GateTest, RandomPauliRotationMerge) {
    UINT n = 5;
    ITYPE dim = 1ULL << n;
    const double eps = 1e-14;

    UINT gate_count = 10;
    UINT max_repeat = 3;
    Random random;
    random.set_seed(2);

    std::vector<UINT> new_pauli_ids = { 0,0,0,1 };
    std::vector<UINT> targets = { 0,1,2,2 };

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		// define states
		QuantumStateGpu state(n, idx), test_state(n, idx);
		Eigen::VectorXcd test_state_eigen(dim);

		for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
			// pick random state and copy to test
			state.set_Haar_random_state();
			test_state.load(&state);
			set_eigen_from_gpu(test_state_eigen, state, dim);

			// check equivalence
			assert_eigen_eq_gpu(test_state_eigen, state, dim, eps);
			assert_gpu_eq_gpu(test_state, state, dim, eps);

			auto merged_gate = gate::Identity(0);
			QuantumGateMatrix* next_merged_gate = NULL;
			QuantumGateBase* new_gate = NULL;
			Eigen::MatrixXcd total_matrix = Eigen::MatrixXcd::Identity(dim, dim);
			//std::cout << "initial state : " << state << std::endl;

			for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {

				// pick random pauli
				UINT new_pauli_id = (random.int32() % 3) + 1;
				UINT target = random.int32() % n;
				double angle = random.uniform() * 3.14159;
				//UINT new_pauli_id = new_pauli_ids[gate_index];
				//UINT target = targets[gate_index];
				if (new_pauli_id == 1) new_gate = gate::RX(target, angle);
				else if (new_pauli_id == 2) new_gate = gate::RY(target, angle);
				else if (new_pauli_id == 3) new_gate = gate::RZ(target, angle);
				else FAIL();

				//std::cout << "***************************************************" << std::endl;
				//std::cout << " ***** Pauli = " << new_pauli_id << " at " << target << std::endl;

				// create new gate with merge
				next_merged_gate = gate::merge(merged_gate, new_gate);
				delete merged_gate;
				merged_gate = next_merged_gate;
				next_merged_gate = NULL;

				// update test state with latest gate
				new_gate->update_quantum_state(&test_state);

				// update eigen state with matrix mul
				auto new_gate_matrix = get_expanded_eigen_matrix_with_identity(target, cos(angle / 2) * ComplexMatrix::Identity(2, 2) + 1.i * sin(angle / 2) * get_eigen_matrix_single_Pauli(new_pauli_id), n);
				total_matrix = new_gate_matrix * total_matrix;
				test_state_eigen = new_gate_matrix * test_state_eigen;

				ComplexMatrix check_mat;
				merged_gate->set_matrix(check_mat);
				if (check_mat.rows() == total_matrix.rows()) {
					for (ITYPE x = 0; x < dim; ++x) {
						for (ITYPE y = 0; y < dim; ++y) {
							ASSERT_NEAR(abs(total_matrix(x, y) - check_mat(x, y)), 0, eps) << (QuantumGateMatrix*)merged_gate << std::endl << "current eigen matrix : \n" << total_matrix << std::endl;
						}
					}
				}

				// dispose picked pauli
				delete new_gate;
			}
			merged_gate->update_quantum_state(&state);
			delete merged_gate;
			// check equivalence
			assert_eigen_eq_gpu(test_state_eigen, state, dim, eps);
			assert_gpu_eq_gpu(test_state, state, dim, eps);
		}
	}
}


TEST(GateTest, RandomUnitaryMerge) {
    UINT n = 5;
    ITYPE dim = 1ULL << n;
    const double eps = 1e-14;

    UINT gate_count = 10;
    UINT max_repeat = 3;
    Random random;
    random.set_seed(2);

    std::vector<UINT> new_pauli_ids = { 0,0,0,1 };
    std::vector<UINT> targets = { 0,1,2,2 };

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		set_device(idx);
		auto stream_ptr = allocate_cuda_stream_host(1, idx);

		// define states
		QuantumStateGpu state(n, idx), test_state(n, idx);
		Eigen::VectorXcd test_state_eigen(dim);

		for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
			// pick random state and copy to test
			state.set_Haar_random_state();
			test_state.load(&state);
			set_eigen_from_gpu(test_state_eigen, state, dim);

			// check equivalence
			assert_eigen_eq_gpu(test_state_eigen, state, dim, eps);
			assert_gpu_eq_gpu(test_state, state, dim, eps);

			auto merged_gate = gate::Identity(0);
			QuantumGateMatrix* next_merged_gate = NULL;
			QuantumGateBase* new_gate = NULL;
			Eigen::MatrixXcd total_matrix = Eigen::MatrixXcd::Identity(dim, dim);
			//std::cout << "initial state : " << state << std::endl;

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
				ComplexMatrix mat = di * get_eigen_matrix_single_Pauli(0) + 1.i * (dx * get_eigen_matrix_single_Pauli(1) + dy * get_eigen_matrix_single_Pauli(2) + dz * get_eigen_matrix_single_Pauli(3));

				auto new_gate = gate::DenseMatrix(target, mat);

				// create new gate with merge
				next_merged_gate = gate::merge(merged_gate, new_gate);
				delete merged_gate;
				merged_gate = next_merged_gate;
				next_merged_gate = NULL;

				// update test state with latest gate
				new_gate->update_quantum_state(&test_state);

				// update eigen state with matrix mul
				auto new_gate_matrix = get_expanded_eigen_matrix_with_identity(target, mat, n);
				total_matrix = new_gate_matrix * total_matrix;
				test_state_eigen = new_gate_matrix * test_state_eigen;

				ComplexMatrix check_mat;
				merged_gate->set_matrix(check_mat);
				if (check_mat.rows() == total_matrix.rows()) {
					for (ITYPE x = 0; x < dim; ++x) {
						for (ITYPE y = 0; y < dim; ++y) {
							ASSERT_NEAR(abs(total_matrix(x, y) - check_mat(x, y)), 0, eps) << (QuantumGateMatrix*)merged_gate << std::endl << "current eigen matrix : \n" << total_matrix << std::endl;
						}
					}
				}

				// dispose picked pauli
				delete new_gate;
			}
			merged_gate->update_quantum_state(&state);
			delete merged_gate;
			// check equivalence
			assert_eigen_eq_gpu(test_state_eigen, state, dim, eps);
			assert_gpu_eq_gpu(test_state, state, dim, eps);
		}
	}
}



TEST(GateTest, RandomUnitaryMergeLarge) {
    UINT n = 5;
    ITYPE dim = 1ULL << n;
    const double eps = 1e-14;

    UINT gate_count = 5;
    UINT max_repeat = 2;
    Random random;
    random.set_seed(2);

    std::vector<UINT> new_pauli_ids = { 0,0,0,1 };
    std::vector<UINT> targets = { 0,1,2,2 };

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		// define states
		QuantumStateGpu state(n, idx), test_state(n, idx);
		Eigen::VectorXcd test_state_eigen(dim);

		for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
			// pick random state and copy to test
			state.set_Haar_random_state();
			test_state.load(&state);
			set_eigen_from_gpu(test_state_eigen, state, dim);

			// check equivalence
			assert_eigen_eq_gpu(test_state_eigen, state, dim, eps);
			assert_gpu_eq_gpu(test_state, state, dim, eps);

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
				ComplexMatrix mat = di * get_eigen_matrix_single_Pauli(0) + 1.i * (dx * get_eigen_matrix_single_Pauli(1) + dy * get_eigen_matrix_single_Pauli(2) + dz * get_eigen_matrix_single_Pauli(3));

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
				ComplexMatrix mat = di * get_eigen_matrix_single_Pauli(0) + 1.i * (dx * get_eigen_matrix_single_Pauli(1) + dy * get_eigen_matrix_single_Pauli(2) + dz * get_eigen_matrix_single_Pauli(3));

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
			merged_gate1->update_quantum_state(&test_state);
			merged_gate2->update_quantum_state(&test_state);

			delete merged_gate;
			delete merged_gate1;
			delete merged_gate2;
			// check equivalence
			assert_gpu_eq_gpu(test_state, state, dim, eps);
		}
	}
}

TEST(GateTest, U3MergeIBMQGate) {
    auto gate1 = gate::U3(0, 0.1, 0.1, 0.1);
    auto gate2 = gate::U3(0, 0.1, 0.1, 0.1);
    auto gate3 = gate::merge(gate1, gate2);
}

TEST(GateTest, ControlMerge) {
    UINT n = 2;
    ITYPE dim = 1ULL << n;
    const double eps = 1e-14;

    {
        auto x0 = gate::X(0);
        auto cx01 = gate::CNOT(0, 1);
        auto res = gate::merge(x0, cx01);
        //std::cout << res << std::endl;

        auto mat_x = get_expanded_eigen_matrix_with_identity(0, get_eigen_matrix_single_Pauli(1), 2);
        auto mat_cx = get_eigen_matrix_full_qubit_CNOT(0, 1, 2);
        auto mat_res = mat_cx * mat_x;
        //std::cout << mat_res << std::endl;

        ComplexMatrix checkmat;
        res->set_matrix(checkmat);
        for (ITYPE x = 0; x < dim; ++x) for (ITYPE y = 0; y < dim; ++y) ASSERT_NEAR(abs(checkmat(x, y) - (mat_res(x, y))), 0, eps) << res << "\n\n" << mat_res << std::endl;
    }

    {
        auto x0 = gate::X(0);
        auto cx01 = gate::CNOT(0, 1);
        auto res = gate::merge(cx01,x0);
        //std::cout << res << std::endl;

        auto mat_x = get_expanded_eigen_matrix_with_identity(0, get_eigen_matrix_single_Pauli(1), 2);
        auto mat_cx = get_eigen_matrix_full_qubit_CNOT(0, 1, 2);
        auto mat_res = mat_x * mat_cx;
        //std::cout << mat_res << std::endl;

        ComplexMatrix checkmat;
        res->set_matrix(checkmat);
        for (ITYPE x = 0; x < dim; ++x) for (ITYPE y = 0; y < dim; ++y) ASSERT_NEAR(abs(checkmat(x, y) - (mat_res(x, y))), 0, eps) << res << "\n\n" << mat_res << std::endl;
    }

    {
        auto x1 = gate::X(1);
        auto cx01 = gate::CNOT(0, 1);
        auto res = gate::merge(x1, cx01);
        //std::cout << res << std::endl;

        auto mat_x = get_expanded_eigen_matrix_with_identity(1, get_eigen_matrix_single_Pauli(1), 2);
        auto mat_cx = get_eigen_matrix_full_qubit_CNOT(0, 1, 2);
        auto mat_res = mat_cx * mat_x;
        //std::cout << mat_res << std::endl;

        ComplexMatrix checkmat;
        res->set_matrix(checkmat);
        for (ITYPE x = 0; x < dim; ++x) for (ITYPE y = 0; y < dim; ++y) ASSERT_NEAR(abs(checkmat(x, y) - (mat_res(x, y))), 0, eps) << res << "\n\n" << mat_res << std::endl;
    }

    {
        auto x1 = gate::X(1);
        auto cx01 = gate::CNOT(0, 1);
        auto res = gate::merge(cx01, x1);
        //std::cout << res << std::endl;

        auto mat_x = get_expanded_eigen_matrix_with_identity(1, get_eigen_matrix_single_Pauli(1), 2);
        auto mat_cx = get_eigen_matrix_full_qubit_CNOT(0, 1, 2);
        auto mat_res = mat_x * mat_cx;
        //std::cout << mat_res << std::endl;

        ComplexMatrix checkmat;
        res->set_matrix(checkmat);
        for (ITYPE x = 0; x < dim; ++x) for (ITYPE y = 0; y < dim; ++y) ASSERT_NEAR(abs(checkmat(x, y) - (mat_res(x, y))), 0, eps) << res << "\n\n" << mat_res << std::endl;
    }

    {
        auto cz01 = gate::CZ(0,1);
        auto cx01 = gate::CNOT(0, 1);
        auto res = gate::merge(cx01, cz01);
        //std::cout << res << std::endl;

        ASSERT_EQ(res->control_qubit_list.size(), 1);
        ASSERT_EQ(res->control_qubit_list[0].index(), 0);
        ComplexMatrix mat_res = 1.i * get_eigen_matrix_single_Pauli(2);


        ComplexMatrix checkmat;
        res->set_matrix(checkmat);
        for (ITYPE x = 0; x < 2; ++x) for (ITYPE y = 0; y < 2; ++y) ASSERT_NEAR(abs(checkmat(x, y) - (mat_res(x, y))), 0, eps) << res << "\n\n" << mat_res << std::endl;
    }

    {
        auto cz10 = gate::CZ(1,0);
        auto cx01 = gate::CNOT(0, 1);
        auto res = gate::merge(cx01, cz10);

        auto mat_cz = get_eigen_matrix_full_qubit_CZ(1, 0,2);
        auto mat_cx = get_eigen_matrix_full_qubit_CNOT(0, 1, 2);
        auto mat_res = mat_cz * mat_cx;

        ComplexMatrix checkmat;
        res->set_matrix(checkmat);
        for (ITYPE x = 0; x < dim; ++x) for (ITYPE y = 0; y < dim; ++y) ASSERT_NEAR(abs(checkmat(x, y) - (mat_res(x, y))), 0, eps) << res << "\n\n" << mat_res << std::endl;
    }


    n = 3;
    dim = 1ULL << n;
    {
        auto x2 = gate::X(2);
        auto cx01 = gate::CNOT(0, 1);
        auto res = gate::merge(x2, cx01);
        //std::cout << res << std::endl;

        auto mat_x = get_expanded_eigen_matrix_with_identity(2, get_eigen_matrix_single_Pauli(1), n);
        auto mat_cx = get_eigen_matrix_full_qubit_CNOT(0, 1, n);
        auto mat_res = mat_cx * mat_x;
        //std::cout << mat_res << std::endl;

        ComplexMatrix checkmat;
        res->set_matrix(checkmat);
        for (ITYPE x = 0; x < dim; ++x) for (ITYPE y = 0; y < dim; ++y) ASSERT_NEAR(abs(checkmat(x, y) - (mat_res(x, y))), 0, eps) << res << "\n\n" << mat_res << std::endl;
    }

    {
        auto x2 = gate::X(2);
        auto cx01 = gate::CNOT(0, 1);
        auto res = gate::merge(cx01, x2);
        //std::cout << res << std::endl;

        auto mat_x = get_expanded_eigen_matrix_with_identity(2, get_eigen_matrix_single_Pauli(1), n);
        auto mat_cx = get_eigen_matrix_full_qubit_CNOT(0, 1, n);
        auto mat_res = mat_x * mat_cx;
        //std::cout << mat_res << std::endl;

        ComplexMatrix checkmat;
        res->set_matrix(checkmat);
        for (ITYPE x = 0; x < dim; ++x) for (ITYPE y = 0; y < dim; ++y) ASSERT_NEAR(abs(checkmat(x, y) - (mat_res(x, y))), 0, eps) << res << "\n\n" << mat_res << std::endl;
    }

}



TEST(GateTest, RandomControlMergeSmall) {
    UINT n = 4;
    ITYPE dim = 1ULL << n;
    const double eps = 1e-14;

    UINT gate_count = 10;
    Random random;

    std::vector<UINT> arr;
    for (UINT i = 0; i < n; ++i) arr.push_back(i);

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		for (gate_count = 1; gate_count < n * 2; ++gate_count) {
			ComplexMatrix mat = ComplexMatrix::Identity(dim, dim);
			QuantumStateGpu state(n, idx), test_state(n, idx);
			ComplexVector test_state_eigen(dim);
			state.set_Haar_random_state();
			test_state.load(&state);
			set_eigen_from_gpu(test_state_eigen, state, dim);
			auto merge_gate1 = gate::Identity(0);
			auto merge_gate2 = gate::Identity(0);

			for (UINT gate_index = 0; gate_index < gate_count; ++gate_index) {
				std::random_shuffle(arr.begin(), arr.end());
				UINT target = arr[0];
				UINT control = arr[1];
				auto new_gate = gate::CNOT(control, target);
				merge_gate1 = gate::merge(merge_gate1, new_gate);

				new_gate->update_quantum_state(&test_state);

				auto cmat = get_eigen_matrix_full_qubit_CNOT(control, target, n);
				mat = cmat * mat;
			}
			merge_gate1->update_quantum_state(&state);
			test_state_eigen = mat * test_state_eigen;

			assert_eigen_eq_gpu(test_state_eigen, state, dim, eps);
			assert_gpu_eq_gpu(test_state, state, dim, eps);
		}
	}
}


TEST(GateTest, RandomControlMergeLarge) {
    UINT n = 4;
    ITYPE dim = 1ULL << n;
    const double eps = 1e-14;

    UINT gate_count = 10;
    Random random;

    std::vector<UINT> arr;
    for (UINT i = 0; i < n; ++i) arr.push_back(i);

	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		for (gate_count = 1; gate_count < n * 2; ++gate_count) {
			ComplexMatrix mat = ComplexMatrix::Identity(dim, dim);
			QuantumStateGpu state(n, idx), test_state(n, idx);
			ComplexVector test_state_eigen(dim);
			state.set_Haar_random_state();
			test_state.load(&state);
			set_eigen_from_gpu(test_state_eigen, state, dim);
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
			merge_gate1->update_quantum_state(&test_state);
			merge_gate2->update_quantum_state(&test_state);
			test_state_eigen = mat * test_state_eigen;

			assert_eigen_eq_gpu(test_state_eigen, state, dim, eps);
			assert_gpu_eq_gpu(test_state, state, dim, eps);
		}
	}
}

TEST(GateTest, ProbabilisticGate) {
    auto gate1 = gate::X(0);
    auto gate2 = gate::X(1);
    auto gate3 = gate::X(2);
    auto prob_gate = gate::Probabilistic({ 0.25,0.25,0.25 }, { gate1, gate2, gate2 });
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		QuantumStateGpu s(3, idx);
		s.set_computational_basis(0);
		prob_gate->update_quantum_state(&s);
	}
    delete gate1;
    delete gate2;
    delete gate3;
    delete prob_gate;
}

TEST(GateTest, CPTPGate) {
    auto gate1 = gate::merge(gate::P0(0), gate::P0(1));
    auto gate2 = gate::merge(gate::P0(0), gate::P1(1));
    auto gate3 = gate::merge(gate::P1(0), gate::P0(1));
    auto gate4 = gate::merge(gate::P1(0), gate::P1(1));

    auto CPTP = gate::CPTP({ gate3, gate2, gate1, gate4 });
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		QuantumStateGpu s(3, idx);
		s.set_computational_basis(0);
		CPTP->update_quantum_state(&s);
		s.set_Haar_random_state();
		CPTP->update_quantum_state(&s);
	}
    delete CPTP;
}

TEST(GateTest, InstrumentGate) {
    auto gate1 = gate::merge(gate::P0(0), gate::P0(1));
    auto gate2 = gate::merge(gate::P0(0), gate::P1(1));
    auto gate3 = gate::merge(gate::P1(0), gate::P0(1));
    auto gate4 = gate::merge(gate::P1(0), gate::P1(1));

    auto Inst = gate::Instrument({ gate3, gate2, gate1, gate4 }, 1);
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		QuantumStateGpu s(3, idx);
		s.set_computational_basis(0);
		Inst->update_quantum_state(&s);
		UINT res1 = s.get_classical_value(1);
		ASSERT_EQ(res1, 2);
		s.set_Haar_random_state();
		Inst->update_quantum_state(&s);
		UINT res2 = s.get_classical_value(1);
	}
    delete Inst;
}

TEST(GateTest, AdaptiveGate) {
    auto x = gate::X(0);
    auto adaptive = gate::Adaptive(x, [](const std::vector<UINT>& vec) { return vec[2] == 1; });
	int ngpus = get_num_device();
	for (int idx = 0; idx < ngpus; ++idx) {
		QuantumStateGpu s(1, idx);
		s.set_computational_basis(0);
		s.set_classical_value(2, 1);
		adaptive->update_quantum_state(&s);
		s.set_classical_value(2, 0);
		adaptive->update_quantum_state(&s);
	}
    delete adaptive;
}

TEST(GateTest, GateAdd) {
    auto g1 = gate::X(0);
    auto g2 = gate::X(0);
    auto g3 = gate::X(1);
    auto g4 = gate::CNOT(0, 1);

    auto a1 = gate::add(g1, g2);
    auto a2 = gate::add(g1, g3);
    auto a3 = gate::add(g1, g4);
    auto a4 = gate::add(g3, g4);
    auto a5 = gate::add(gate::P0(0), gate::P1(0));
    auto a6 = gate::add(gate::merge(gate::P0(0), gate::P0(1)), gate::merge(gate::P1(0), gate::P1(1)));
    // TODO assert matrix element
}
