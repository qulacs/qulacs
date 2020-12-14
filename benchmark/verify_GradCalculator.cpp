#include <cppsim/observable.hpp>
#include <vqcsim/parametric_circuit.hpp>
#include <cppsim/state.hpp>
#include <vqcsim/GradCalculator.hpp>
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>

int main() {    
    omp_set_num_threads(10);
    printf("使用可能な最大スレッド数：%d\n", omp_get_max_threads());
    
    std::chrono::system_clock::time_point start, end;

    start = std::chrono::system_clock::now();

    srand(0);
    unsigned int n = 20;
    Observable observable(n);
    std::string Pauli_string = "";
    for(int i = 0;i < n;++i){
        double coef = (float)rand() / (float)RAND_MAX;
        std::string Pauli_string = "Z ";
        Pauli_string += std::to_string(i);
        observable.add_operator(coef, Pauli_string.c_str());
    }

    ParametricQuantumCircuit circuit(n);
    
    for(int depth = 0;depth < 5;++depth){
        for(int i = 0;i < n;++i){
            circuit.add_parametric_RX_gate(i,0);
            circuit.add_parametric_RZ_gate(i,0);
        }
        
        for(int i = 0;i + 1 < n;i += 2){
            circuit.add_CNOT_gate(i,i+1);
        }
        
        for(int i = 1;i + 1 < n;i += 2){
            circuit.add_CNOT_gate(i,i+1);
        }
    }
    
    GradCalculator hoge;
    std::vector<std::complex<double>> ans = hoge.calculate_grad(circuit,observable,rand());
    
    end = std::chrono::system_clock::now();

    double msec = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
   
    std::cout << "GradCalculator msec: " << msec << " msec"<< std::endl;
    for(int i = 0;i < ans.size();++i){
        std::cout << ans[i] << std::endl;
    }
    return 0;
}