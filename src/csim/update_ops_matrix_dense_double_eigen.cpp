
#define EIGEN_DONT_PARALLELIZE
#ifndef _MSC_VER
#include "update_ops_cpp.hpp"
extern "C" {
#include "utility.h"
#include "update_ops.h"
}
#else
#include "update_ops_cpp.hpp"
#include "update_ops.h"
#include "utility.h"
#endif
#include <Eigen/Core>

void double_qubit_dense_matrix_gate(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim) {
	double_qubit_dense_matrix_gate_c(target_qubit_index1, target_qubit_index2, matrix, state, dim);
}

void double_qubit_dense_matrix_gate(UINT target_qubit_index1, UINT target_qubit_index2, const Eigen::Matrix4cd& eigen_matrix, CTYPE *state, ITYPE dim) {
	double_qubit_dense_matrix_gate_eigen(target_qubit_index1, target_qubit_index2, eigen_matrix, state, dim);
}

void double_qubit_dense_matrix_gate_eigen(UINT target_qubit_index1, UINT target_qubit_index2, const Eigen::Matrix4cd& eigen_matrix, CTYPE *state, ITYPE dim) {
	// target mask

	const UINT min_qubit_index = get_min_ui(target_qubit_index1, target_qubit_index2);
	const UINT max_qubit_index = get_max_ui(target_qubit_index1, target_qubit_index2);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	const ITYPE target_mask1 = 1ULL << target_qubit_index1;
	const ITYPE target_mask2 = 1ULL << target_qubit_index2;
	std::complex<double>* eigen_state = reinterpret_cast<std::complex<double>*>(state);

	// loop variables
	const ITYPE loop_dim = dim / 4;
	ITYPE state_index;

	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create index
		ITYPE basis_0 = (state_index&low_mask)
			+ ((state_index&mid_mask) << 1)
			+ ((state_index&high_mask) << 2);

		// gather index
		ITYPE basis_1 = basis_0 + target_mask1;
		ITYPE basis_2 = basis_0 + target_mask2;
		ITYPE basis_3 = basis_1 + target_mask2;

		// fetch values
		Eigen::Vector4cd vec(state[basis_0], state[basis_1], state[basis_2], state[basis_3]);
		vec = eigen_matrix * vec;
		eigen_state[basis_0] = vec[0];
		eigen_state[basis_1] = vec[1];
		eigen_state[basis_2] = vec[2];
		eigen_state[basis_3] = vec[3];
	}
}
