
#ifndef _MSC_VER
extern "C" {
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
}
#else
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
#endif
#include <csim/update_ops_cpp.hpp>

#include <cppsim/type.hpp>
#include <cppsim/utility.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <functional>
#include <algorithm>

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



int main() {
	const UINT min_qubit_count = 2;
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
	std::ofstream fout(fname,std::ios::out);

	std::function<double(UINT, double, UINT, std::function<void(UINT, Timer&)>)> benchmark_func = [](UINT n, double timeout_, UINT max_repeat_, std::function<void(UINT, Timer&)> func) {
		UINT repeat = 0;
		Timer timer;
		timer.reset();
		timer.temporal_stop();

		do {
			func(n, timer);
			repeat++;
		} while (timer.elapsed() < timeout_ && repeat <= max_repeat_);
		return timer.elapsed() / repeat;
	};

	std::function<double(UINT, double, UINT, std::function<void(UINT, CTYPE*,Timer&)>)> benchmark_state_ops_func = [](UINT n, double timeout_, UINT max_repeat_, std::function<void(UINT,CTYPE*, Timer&)> func) {
		UINT repeat = 0;
		Timer timer;
		timer.reset();
		timer.temporal_stop();
		ITYPE dim = 1ULL << n;
		CTYPE* state_ptr = allocate_quantum_state(dim);
		initialize_Haar_random_state(state_ptr,dim);
		do {
			func(n, state_ptr,timer);
			repeat++;
		} while (timer.elapsed() < timeout_ && repeat <= max_repeat_);
		release_quantum_state(state_ptr);
		return timer.elapsed() / repeat;
	};

	std::function<void(UINT, Timer&)> func;
	std::function<void(UINT, CTYPE*, Timer&)> state_ops_func;
	std::string name;
	double elapsed;
	for (UINT qubit_count = min_qubit_count; qubit_count < max_qubit_count; ++qubit_count) {
		func = [](UINT n, Timer& timer) {
			CTYPE* state_ptr;
			ITYPE dim = 1ULL << n;
			timer.temporal_resume();
			state_ptr = allocate_quantum_state(dim);
			timer.temporal_stop();
			release_quantum_state(state_ptr);
		};
		name = "memory_allocate";
		elapsed = benchmark_func(qubit_count, timeout, max_repeat, func);
		std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
		fout << name << " " << qubit_count << " " << elapsed << std::endl;

		func = [](UINT n, Timer& timer) {
			CTYPE* state_ptr;
			ITYPE dim = 1ULL << n;
			state_ptr = allocate_quantum_state(dim);
			timer.temporal_resume();
			release_quantum_state(state_ptr);
			timer.temporal_stop();
		};
		name = "memory_release";
		elapsed = benchmark_func(qubit_count, timeout, max_repeat, func);
		std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
		fout << name << " " << qubit_count << " " << elapsed << std::endl;

		func = [](UINT n, Timer& timer) {
			CTYPE* state_ptr;
			ITYPE dim = 1ULL << n;
			state_ptr = allocate_quantum_state(dim);
			timer.temporal_resume();
			initialize_quantum_state(state_ptr, dim);
			timer.temporal_stop();
			release_quantum_state(state_ptr);
		};
		name = "initialize_state_zero";
		elapsed = benchmark_func(qubit_count, timeout, max_repeat, func);
		std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
		fout << name << " " << qubit_count << " " << elapsed << std::endl;

		func = [](UINT n, Timer& timer) {
			CTYPE* state_ptr;
			ITYPE dim = 1ULL << n;
			state_ptr = allocate_quantum_state(dim);
			timer.temporal_resume();
			initialize_Haar_random_state(state_ptr, dim);
			timer.temporal_stop();
			release_quantum_state(state_ptr);
		};
		name = "initialize_state_random";
		elapsed = benchmark_func(qubit_count, timeout, max_repeat, func);
		std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
		fout << name << " " << qubit_count << " " << elapsed << std::endl;


		state_ops_func = [](UINT n,CTYPE* state, Timer& timer) {
			ITYPE dim = 1ULL << n;
			timer.temporal_resume();
			X_gate(0, state, dim);
			timer.temporal_stop();
		};
		name = "X_gate";
		elapsed = benchmark_state_ops_func(qubit_count, timeout, max_repeat, state_ops_func);
		std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
		fout << name << " " << qubit_count << " " << elapsed << std::endl;

		state_ops_func = [](UINT n, CTYPE* state, Timer& timer) {
			ITYPE dim = 1ULL << n;
			timer.temporal_resume();
			Y_gate(0, state, dim);
			timer.temporal_stop();
		};
		name = "Y_gate";
		elapsed = benchmark_state_ops_func(qubit_count, timeout, max_repeat, state_ops_func);
		std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
		fout << name << " " << qubit_count << " " << elapsed << std::endl;

		state_ops_func = [](UINT n, CTYPE* state, Timer& timer) {
			ITYPE dim = 1ULL << n;
			timer.temporal_resume();
			Z_gate(0, state, dim);
			timer.temporal_stop();
		};
		name = "Z_gate";
		elapsed = benchmark_state_ops_func(qubit_count, timeout, max_repeat, state_ops_func);
		std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
		fout << name << " " << qubit_count << " " << elapsed << std::endl;

		state_ops_func = [](UINT n, CTYPE* state, Timer& timer) {
			ITYPE dim = 1ULL << n;
			timer.temporal_resume();
			H_gate(0, state, dim);
			timer.temporal_stop();
		};
		name = "H_gate";
		elapsed = benchmark_state_ops_func(qubit_count, timeout, max_repeat, state_ops_func);
		std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
		fout << name << " " << qubit_count << " " << elapsed << std::endl;

		state_ops_func = [](UINT n, CTYPE* state, Timer& timer) {
			ITYPE dim = 1ULL << n;
			timer.temporal_resume();
			CNOT_gate(0,1, state, dim);
			timer.temporal_stop();
		};
		name = "CNOT_gate";
		elapsed = benchmark_state_ops_func(qubit_count, timeout, max_repeat, state_ops_func);
		std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
		fout << name << " " << qubit_count << " " << elapsed << std::endl;

		state_ops_func = [](UINT n, CTYPE* state, Timer& timer) {
			ITYPE dim = 1ULL << n;
			timer.temporal_resume();
			CZ_gate(0, 1, state, dim);
			timer.temporal_stop();
		};
		name = "CZ_gate";
		elapsed = benchmark_state_ops_func(qubit_count, timeout, max_repeat, state_ops_func);
		std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
		fout << name << " " << qubit_count << " " << elapsed << std::endl;

		state_ops_func = [](UINT n, CTYPE* state, Timer& timer) {
			ITYPE dim = 1ULL << n;
			timer.temporal_resume();
			SWAP_gate(0, 1, state, dim);
			timer.temporal_stop();
		};
		name = "SWAP_gate";
		elapsed = benchmark_state_ops_func(qubit_count, timeout, max_repeat, state_ops_func);
		std::cout << name << " " << qubit_count << " " << elapsed << std::endl;
		fout << name << " " << qubit_count << " " << elapsed << std::endl;

		
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
	}
	fout.close();
}

