#include <iostream>
#include <complex>
#include <Eigen/Dense>

// Incluye tus propias cabeceras:
#include <cppsim/gate.hpp>         // debe incluir QuantumGateBase
#include <cppsim/gate_named_one.hpp>  // o como se llame tu archivo con ClsOneQubitGate
#include <cppsim/gate_named_two.hpp>


int main() {
/*     QuantumGate_OneQubit x_gate;
    x_gate.XGateinit(0); // Inicializa la puerta X en el qubit 0

    ComplexMatrix matrix_x;
    x_gate.set_matrix(matrix_x);

    std::cout << "Matriz de la puerta X:" << std::endl;
    std::cout << matrix_x << std::endl;

    ///////////////////////////

    QuantumGate_TwoQubit swap_gate;
    swap_gate.SWAPGateinit(0,1); // Inicializa la puerta X en el qubit 0

    ComplexMatrix matrix_swap;
    swap_gate.set_matrix(matrix_swap);

    std::cout << "Matriz de la puerta SWAP:" << std::endl;
    std::cout << matrix_swap << std::endl; */

    ///////////////////////////

    QuantumGate_TwoQubit ecr_gate;
    ecr_gate.ECRGateinit(0,1); // Inicializa la puerta X en el qubit 0 y 1

    ComplexMatrix matrix_ecr;
    ecr_gate.set_matrix(matrix_ecr);

    std::cout << "Matriz de la puerta ECR:" << std::endl;
    std::cout << matrix_ecr << std::endl;

    //////////////////////////////////////////////////////////////
    
    std::cout << "\n=== Prueba de dm_ECR_gate ===" << std::endl;

    // 1. Crear una matriz de densidad 4x4 (para 2 qubits), estado |00>
    ITYPE dim = 4;
    Eigen::MatrixXcd rho = Eigen::MatrixXcd::Zero(dim, dim);
    rho(0, 0) = 1.0;  // |00><00|

    // 2. Copiar rho a un arreglo CTYPE* (formato usado por dm_ECR_gate)
    std::vector<CTYPE> rho_array(dim * dim);
    for (ITYPE i = 0; i < dim; ++i) {
        for (ITYPE j = 0; j < dim; ++j) {
            rho_array[i * dim + j] = rho(i, j);
        }
    }

    // 3. Llamar a dm_ECR_gate (debes tener la función declarada externamente)
    dm_ECR_gate(0, 1, rho_array.data(), dim);  // Aplica ECR sobre qubits 0 y 1

    // 4. Reconstruir la matriz de densidad resultante
    Eigen::MatrixXcd rho_after(dim, dim);
    for (ITYPE i = 0; i < dim; ++i) {
        for (ITYPE j = 0; j < dim; ++j) {
            rho_after(i, j) = rho_array[i * dim + j];
        }
    }

    // 5. Ver resultado
    std::cout << "Matriz de densidad después de aplicar ECR:" << std::endl;
    std::cout << rho_after << std::endl;

    // 6. Comparar con resultado esperado: U * rho * U^dagger
    ComplexMatrix U = matrix_ecr;
    Eigen::MatrixXcd expected_rho = U * rho * U.adjoint();

    std::cout << "Resultado esperado (U * rho * U^†):" << std::endl;
    std::cout << expected_rho << std::endl;

    // 7. Comparación
    double tol = 1e-10;
    if ((expected_rho - rho_after).norm() < tol) {
        std::cout << "✅ dm_ECR_gate funciona correctamente." << std::endl;
    } else {
        std::cout << "❌ dm_ECR_gate NO produce el resultado esperado." << std::endl;
    }




    return 0;
}


// g++ -O2 -I ../include -L ../lib proba_matrices.cpp -o proba_dm -lvqcsim_static -lcppsim_static -lcsim_static -fopenmp