#include <mpi.h>
#include <omp.h>

#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>
#include <fstream>
#include <iostream>
#include <mpisim/GradCalculatorMPI.hpp>
#include <vqcsim/parametric_circuit.hpp>

int main(int argc, char **argv) {
    omp_set_num_threads(10);
    printf("使用可能な最大スレッド数：%d\n", omp_get_max_threads());
    MPI_Init(&argc, &argv);

    auto start = MPI_Wtime();

    srand(0);
    unsigned int n = 20;
    Observable observable(n);
    std::string Pauli_string = "";
    for (int i = 0; i < n; ++i) {
        double coef = (float)rand() / (float)RAND_MAX;
        std::string Pauli_string = "Z ";
        Pauli_string += std::to_string(i);
        observable.add_operator(coef, Pauli_string.c_str());
    }

    ParametricQuantumCircuit circuit(n);

    for (int depth = 0; depth < 5; ++depth) {
        for (int i = 0; i < n; ++i) {
            circuit.add_parametric_RX_gate(i, 0);
            circuit.add_parametric_RZ_gate(i, 0);
        }

        for (int i = 0; i + 1 < n; i += 2) {
            circuit.add_CNOT_gate(i, i + 1);
        }

        for (int i = 1; i + 1 < n; i += 2) {
            circuit.add_CNOT_gate(i, i + 1);
        }
    }

    GradCalculatorMPI hoge;
    std::vector<std::complex<double>> ans =
        hoge.calculate_grad(circuit, observable, rand());
    auto end = MPI_Wtime();
    auto dur = end - start;
    auto msec = dur;
    std::cout << "GradCalculatorMPI msec: " << msec << " msec" << std::endl;
    for (int i = 0; i < ans.size(); ++i) {
        std::cout << ans[i] << std::endl;
    }

    MPI_Finalize();
    return 0;
}