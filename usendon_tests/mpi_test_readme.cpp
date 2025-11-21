#include <iostream>
#include <complex>
#include <mpi.h>

#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int mpirank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

    const unsigned int nqubits = 4;
    // estado que puede usar multi-cpu si está disponible
    QuantumState state(nqubits, 1);
    // generar estado aleatorio (semilla fija para reproducibilidad)
    state.set_Haar_random_state(2023);

    // Creamos un state en single-cpu para poder imprimir el vector completo
    QuantumState ref_state(nqubits);
    ref_state.load(&state); // copia completa a single-cpu

    if (mpirank == 0) {
        const auto* vec = ref_state.data_cpp(); // puntero a std::complex<double>
        const std::size_t dim = 1ULL << nqubits;
        std::cout << "Estado inicial (amplitudes):" << std::endl;
        for (std::size_t i = 0; i < dim; ++i) {
            std::cout << i << ": (" << vec[i].real() << ", " << vec[i].imag() << ")\n";
        }
        std::cout << std::endl;
    }

    // Construimos circuito sencillo: X en qubit 0, luego un merge(CNOT(0,1), Y(1)), y RX en 1
    QuantumCircuit circuit(nqubits);
    circuit.add_ECR_gate(2,3);


    // aplicar circuito al estado paralelo/multi-cpu (o single si no hay multi)
    circuit.update_quantum_state(&state);

    // copiar a ref_state para poder imprimir el vector completo en el rank 0
    ref_state.load(&state);

    if (mpirank == 0) {
        const auto* vec = ref_state.data_cpp();
        const std::size_t dim = 1ULL << nqubits;
        std::cout << "Estado final (amplitudes):" << std::endl;
        for (std::size_t i = 0; i < dim; ++i) {
            std::cout << i << ": (" << vec[i].real() << ", " << vec[i].imag() << ")\n";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
