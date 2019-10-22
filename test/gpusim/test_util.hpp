#pragma once

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <gpusim/util_func.h>
#include "../util/util.h"

static Eigen::VectorXcd copy_cpu_from_gpu(void* gpu_state, ITYPE dim) {
	CPPCTYPE* cpu_state = (CPPCTYPE*)malloc(sizeof(CPPCTYPE)*dim);
	Eigen::VectorXcd state(dim);
	get_quantum_state_host(gpu_state, cpu_state, dim);
	for (ITYPE i = 0; i < dim; ++i) state[i] = cpu_state[i];
	free(cpu_state);
	return state;
}
static void state_equal_gpu(void* state, const Eigen::VectorXcd& test_state, ITYPE dim, std::string gate_string) {
	const double eps = 1e-14;
	Eigen::VectorXcd vec = copy_cpu_from_gpu(state, dim);
	for (ITYPE ind = 0; ind < dim; ++ind) {
		ASSERT_NEAR(abs(vec[ind] - test_state[ind]), 0, eps)
			<< gate_string << " at " << ind << std::endl
			<< "Eigen : " << test_state.transpose() << std::endl
			<< "GPUSIM : " << copy_cpu_from_gpu(state, dim) << std::endl;
	}
}
