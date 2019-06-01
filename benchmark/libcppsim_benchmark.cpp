
#include <cppsim/type.hpp>
#include <cppsim/utility.hpp>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/gate_matrix.hpp>
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
	const UINT min_qubit_count = 6;
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

	std::function<double(UINT, double, UINT, std::function<void(UINT, QuantumState&,Timer&)>)> benchmark_state_ops_func = [](UINT n, double timeout_, UINT max_repeat_, std::function<void(UINT,QuantumState&, Timer&)> func) {
		UINT repeat = 0;
		Timer timer;
		timer.reset();
		timer.temporal_stop();

		ITYPE dim = 1ULL << n;
		QuantumState state = QuantumState(n);
		do {
			func(n, state,timer);
			repeat++;
		} while (timer.elapsed() < timeout_ && repeat <= max_repeat_);
		return timer.elapsed() / repeat;
	};

	std::function<void(UINT, Timer&)> func;
	std::function<void(UINT, CTYPE*, Timer&)> state_ops_func;
	std::string name;
	for (UINT qubit_count = min_qubit_count; qubit_count < max_qubit_count; ++qubit_count) {
		
		for(UINT k=1;k<=max_dense_qubit_count;++k){
			ITYPE matrix_dim = 1ULL << k;
			ComplexMatrix matrix = ComplexMatrix::Identity(matrix_dim,matrix_dim);
			int n = qubit_count;
			QuantumState state(n);
			QuantumCircuit circuit(n);
			std::vector<unsigned int> targets(k,0);
			for(UINT m=0;m<k;++m) targets[m]=m;

			for(int i=0;i<max_repeat;++i){
				circuit.add_dense_matrix_gate(targets,matrix);
			}
			Timer timer;
			timer.reset();
			circuit.update_quantum_state(&state);
			double elapsed = timer.elapsed()/max_repeat;
			std::cout << qubit_count << " " << k << " " << elapsed << std::endl;
		}
	}
	fout.close();
	return 0;
}

