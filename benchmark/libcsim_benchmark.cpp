
#ifndef _MSC_VER
extern "C" {
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
}
#else
#include <csim/memory_ops.h>
#include <csim/init_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
#endif
#include <csim/update_ops_cpp.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <functional>
#include <algorithm>
#include <chrono>

#ifdef _MSC_VER
#include <intrin.h>
std::string get_cpu_name();
std::string get_cpu_name() {
	int CPUInfo[4] = { -1 };
	__cpuid(CPUInfo, 0x80000000);
	unsigned int nExIds = CPUInfo[0];
	char CPUBrandString[0x40] = { 0 };
	for (unsigned int i = 0x80000000; i <= nExIds; ++i){
		__cpuid(CPUInfo, i);
		if (i == 0x80000002)
			memcpy(CPUBrandString,CPUInfo,sizeof(CPUInfo));
		else if (i == 0x80000003)
			memcpy(CPUBrandString + 16,CPUInfo,sizeof(CPUInfo));
		else if (i == 0x80000004)
			memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
	}
	return std::string(CPUBrandString);
}
#else
#include <cpuid.h>
std::string get_cpu_name();
std::string get_cpu_name(){
	char CPUBrandString[0x40];
	unsigned int CPUInfo[4] = {0,0,0,0};
	
	__cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
	unsigned int nExIds = CPUInfo[0];
	
	memset(CPUBrandString, 0, sizeof(CPUBrandString));
	
	for (unsigned int i = 0x80000000; i <= nExIds; ++i){
		__cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
	
		if (i == 0x80000002)
			memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
		else if (i == 0x80000003)
			memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
		else if (i == 0x80000004)
			memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
	}
	return std::string(CPUBrandString);
}
#endif

class Timer {
private:
	std::chrono::system_clock::time_point last;
	long long stock;
	bool is_stop;
public:
	Timer() {
		reset();
		is_stop = false;
	}
	void reset() {
		stock = 0;
		last = std::chrono::system_clock::now();
	}
	double elapsed() {
		if (is_stop) return stock * 1e-6;
		else {
			auto duration = std::chrono::system_clock::now() - last;
			return (stock + std::chrono::duration_cast<std::chrono::microseconds>(duration).count())*1e-6;
		}
	}
	void temporal_stop() {
		if (!is_stop) {
			auto duration = std::chrono::system_clock::now() - last;
			stock += std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
			is_stop = true;
		}
	}
	void temporal_resume() {
		if (is_stop) {
			last = std::chrono::system_clock::now();
			is_stop = false;
		}
	}
};


double timeout = 0.5;
UINT max_repeat = 1000000;
UINT firstrun = 1000;

double benchmark_memory_allocate(UINT n) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	do {
		timer.temporal_resume();
		state = allocate_quantum_state(dim);
		timer.temporal_stop();
		release_quantum_state(state);
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	return timer.elapsed() / repeat;
};

double benchmark_memory_release(UINT n) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	do {
		state = allocate_quantum_state(dim);
		timer.temporal_resume();
		release_quantum_state(state);
		timer.temporal_stop();
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	return timer.elapsed() / repeat;
};

double benchmark_state_func(UINT n, std::function<void(CTYPE*, ITYPE)>func) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	state = allocate_quantum_state(dim);
	do {
		timer.temporal_resume();
		func(state, dim);
		timer.temporal_stop();
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	release_quantum_state(state);
	return timer.elapsed() / repeat;
};

double benchmark_state_init(UINT n) {
	return benchmark_state_func(n, initialize_quantum_state);
}
double benchmark_state_randomize(UINT n) {
	return benchmark_state_func(n, initialize_Haar_random_state);
}

template<typename T>
double benchmark_state_stat_func(UINT n, std::function<T(CTYPE*, ITYPE)>func) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state);
	do {
		timer.temporal_resume();
		func(state, dim);
		timer.temporal_stop();
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	release_quantum_state(state);
	return timer.elapsed() / repeat;
};

double benchmark_state_norm(UINT n) {
	return benchmark_state_func(n, state_norm);
}

double benchmark_gate_single_func(UINT n, UINT target, std::function<void(UINT, CTYPE*, ITYPE)>func) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	CTYPE* state;
	double elapsed;
	state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state,dim);
	timer.reset();
	do {
		func(target, state, dim);
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	elapsed = timer.elapsed();
	elapsed /= repeat;
	release_quantum_state(state);
	return elapsed;
};

double benchmark_gate_double_func(UINT n, UINT target1, UINT target2, std::function<void(UINT, UINT, CTYPE*, ITYPE)>func) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state,dim);
	do {
		timer.temporal_resume();
		func(target1, target2, state, dim);
		timer.temporal_stop();
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	release_quantum_state(state);
	return timer.elapsed() / repeat;
};

double benchmark_gate_dense_single(UINT n, UINT target) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	CTYPE matrix[4] = { 1.0, 0.0, 0.0, 1.0 };
	do {
		timer.temporal_resume();
		single_qubit_dense_matrix_gate(target, matrix, state, dim);
		timer.temporal_stop();
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	release_quantum_state(state);
	return timer.elapsed() / repeat;
};


double benchmark_gate_dense_two(UINT n, UINT t1, UINT t2) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	CTYPE matrix[16];
	for (UINT i = 0; i < 4; ++i) matrix[4 * i + i] = 1.;
	do {
		timer.temporal_resume();
		double_qubit_dense_matrix_gate(t1,t2, matrix, state, dim);
		timer.temporal_stop();
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	release_quantum_state(state);
	return timer.elapsed() / repeat;
};


double benchmark_gate_dense_two_eigen(UINT n, UINT t1, UINT t2) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	Eigen::Matrix4cd matrix;
	for (UINT i = 0; i < 4; ++i) matrix(i,i) = 1.;
	do {
		timer.temporal_resume();
		double_qubit_dense_matrix_gate_eigen(t1, t2, matrix, state, dim);
		timer.temporal_stop();
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	release_quantum_state(state);
	return timer.elapsed() / repeat;
};


double benchmark_gate_dense_three(UINT n, UINT t1, UINT t2, UINT t3) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	CTYPE matrix[64];
	UINT target[3] = { t1,t2,t3 };
	for (UINT i = 0; i < 8; ++i) matrix[8 * i + i] = 1.;
	do {
		timer.temporal_resume();
		multi_qubit_dense_matrix_gate(target, 3, matrix, state, dim);
		timer.temporal_stop();
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	release_quantum_state(state);
	return timer.elapsed() / repeat;
};


double benchmark_gate_diag_single(UINT n, UINT target) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	CTYPE matrix[2] = { 1.0, 1.0 };
	do {
		timer.temporal_resume();
		single_qubit_diagonal_matrix_gate(target, matrix, state, dim);
		timer.temporal_stop();
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	release_quantum_state(state);
	return timer.elapsed() / repeat;
};

double benchmark_gate_phase_single(UINT n, UINT target) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	CTYPE phase = 1.0;
	do {
		timer.temporal_resume();
		single_qubit_phase_gate(target, phase, state, dim);
		timer.temporal_stop();
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	release_quantum_state(state);
	return timer.elapsed() / repeat;
};


double benchmark_gate_single_control_single_target(UINT n, UINT control, UINT target) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	CTYPE matrix[4] = { 1.0, 0.0, 0.0, 1.0 };
	do {
		timer.temporal_resume();
		single_qubit_control_single_qubit_dense_matrix_gate(control, 0, target, matrix, state, dim);
		timer.temporal_stop();
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	release_quantum_state(state);
	return timer.elapsed() / repeat;
};

double benchmark_gate_multi_control_single_target(UINT n, UINT c1, UINT c2, UINT target) {
	UINT repeat = 0;
	ITYPE dim = (1ULL << n);
	Timer timer;
	timer.reset();
	timer.temporal_stop();
	CTYPE* state;
	state = allocate_quantum_state(dim);
	initialize_Haar_random_state(state, dim);
	CTYPE matrix[4] = { 1.0, 0.0, 0.0, 1.0 };
	UINT control[2] = { c1,c2 };
	UINT cvalue[2] = { 0,1 };
	do {
		timer.temporal_resume();
		multi_qubit_control_single_qubit_dense_matrix_gate(control, cvalue, 2, target, matrix, state, dim);
		timer.temporal_stop();
		repeat++;
	} while (timer.elapsed() < timeout && repeat <= max_repeat);
	release_quantum_state(state);
	return timer.elapsed() / repeat;
};

void show(std::string name, UINT qubit_count, double elapsed_time, std::string filename) {
	std::cout << std::fixed << std::setw(20) << name << std::setw(5) << qubit_count << " " << std::setw(20) << std::setprecision(2) << elapsed_time*1e9 << " ns" << std::endl;
	std::ofstream fout(filename, std::ios::app);
	fout << std::fixed << std::setw(20) << name << std::setw(5) << qubit_count << " " << std::setw(20) << std::setprecision(2) << elapsed_time * 1e9 << " ns" << std::endl;
	fout.close();
	//fout << name << " " << qubit_count << " " << elapsed << std::endl;
}

int main() {
	const UINT min_qubit_count = 5;
	const UINT max_qubit_count = 25;
	const double timeout = 0.1;
	const UINT max_repeat = 10;
	const UINT max_dense_qubit_count = 5;

	#ifdef _MSC_VER
	std::string compiler_name = "msvc";
	#elif __GNUG__
	std::string compiler_name = "gcc";
	#elif __clang__
	std::string compiler_name = "clang";
	#else
	std::string compiler_name = "unknowncompiler";
	#endif
	
	std::string fname = "bench_" + compiler_name + "_" + get_cpu_name() + ".txt";
	std::ofstream fout(fname, std::ios::out);
	fout.close();
	for (UINT qubit_count = min_qubit_count; qubit_count < max_qubit_count; ++qubit_count) {

		//show("memory allocate", qubit_count, benchmark_memory_allocate(qubit_count), fname);
		//show("memory release", qubit_count, benchmark_memory_release(qubit_count), fname);

		//show("state initialize", qubit_count, benchmark_state_init(qubit_count), fname);
		//show("state randomize", qubit_count, benchmark_state_randomize(qubit_count), fname);

		//show("gate X_0", qubit_count, benchmark_gate_single_func(qubit_count, 0, X_gate), fname);
		//show("gate X_1", qubit_count, benchmark_gate_single_func(qubit_count, 1, X_gate), fname);
		//show("gate X_3", qubit_count, benchmark_gate_single_func(qubit_count, 3, X_gate), fname);
		//show("gate Y_3", qubit_count, benchmark_gate_single_func(qubit_count, 3, Y_gate), fname);
		//show("gate Z_3", qubit_count, benchmark_gate_single_func(qubit_count, 3, Z_gate), fname);
		//show("gate H_3", qubit_count, benchmark_gate_single_func(qubit_count, 3, H_gate), fname);
		//show("gate P0_3", qubit_count, benchmark_gate_single_func(qubit_count, 3, P0_gate), fname);

		//show("gate CX_2,3", qubit_count, benchmark_gate_double_func(qubit_count, 2, 3, CNOT_gate), fname);
		//show("gate CZ_2,3", qubit_count, benchmark_gate_double_func(qubit_count, 2, 3, CZ_gate), fname);
		//show("gate SWAP_2,3", qubit_count, benchmark_gate_double_func(qubit_count, 2, 3, SWAP_gate), fname);
		//show("gate diag_3", qubit_count, benchmark_gate_diag_single(qubit_count, 3), fname);
		//show("gate phase_3", qubit_count, benchmark_gate_phase_single(qubit_count, 3), fname);

		//show("gate single_3", qubit_count, benchmark_gate_dense_single(qubit_count, 3), fname);

		//show("gate single_control_single_target_2,3", qubit_count, benchmark_gate_single_control_single_target(qubit_count, 2, 3), fname);
		//show("gate multi_control_single_target_2,3", qubit_count, benchmark_gate_multi_control_single_target(qubit_count, 1,2, 3), fname);

		//show("gate two_2,3", qubit_count, benchmark_gate_dense_two(qubit_count, 2,3), fname);
		show("gate two_eigen_2,3", qubit_count, benchmark_gate_dense_two_eigen(qubit_count, 2, 3), fname);
		//show("gate three_1,2,3", qubit_count, benchmark_gate_dense_three(qubit_count, 1,2,3), fname);


		/*
		for(UINT k=1;k<=max_dense_qubit_count;++k){
			ITYPE matrix_dim = 1ULL << k;
			ComplexMatrix matrix = ComplexMatrix::Identity(matrix_dim,matrix_dim);
			CTYPE* ptr = (CTYPE*)matrix.data();
			UINT* targets = (UINT*)calloc(sizeof(UINT),k);
			for(UINT m=0;m<k;++m) targets[m]=m;

			state_ops_func = [&](UINT n, CTYPE* state, Timer& timer) {
				ITYPE dim = 1ULL << n;

				timer.temporal_resume();
				multi_qubit_dense_matrix_gate(targets,k,ptr, state, dim);
				timer.temporal_stop();
			};
			name = "dense_"+std::to_string(k)+"qubit_gate";
			elapsed = benchmark_state_ops_func(qubit_count, timeout, max_repeat, state_ops_func);
			std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
			fout << name << " " << qubit_count << " " << elapsed << std::endl;
			free(targets);
		}
		*/

		/*
		for (UINT k = 1; k <= max_dense_qubit_count; ++k) {
			ITYPE matrix_dim = 1ULL << k;
			ComplexMatrix matrix = ComplexMatrix::Identity(matrix_dim, matrix_dim);
			CTYPE* ptr = (CTYPE*)matrix.data();
			UINT* targets = (UINT*)calloc(sizeof(UINT), k);
			for (UINT m = 0; m < k; ++m) targets[m] = m;

			state_ops_func = [&](UINT n, CTYPE* state, Timer& timer) {
				ITYPE dim = 1ULL << n;

				timer.temporal_resume();
				multi_qubit_dense_matrix_gate_eigen(targets, k, matrix, state, dim);
				timer.temporal_stop();
			};
			name = "dense_" + std::to_string(k) + "qubit_gate_eigen";
			elapsed = benchmark_state_ops_func(qubit_count, timeout, max_repeat, state_ops_func);
			std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
			fout << name << " " << qubit_count << " " << elapsed << std::endl;
			free(targets);
		}
		*/
	}
}

