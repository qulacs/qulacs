#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

int main(int argc, char *argv[]) {
    int mpirank, mpisize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	const UINT global_nqubits = (UINT)std::log2(mpisize);

    int nqubits = 4;
    //QuantumState ref_state(nqubits); // use single cpu
    QuantumState state(nqubits, 1); // use multi_cpu if possible
    state.set_Haar_random_state(2023);
    //ref_state.load(&state);
    if (mpirank == 0) {
		std::string device_name = state.get_device_name();
		std::cout << "Device name of the state vector: " << device_name << std::endl;
        if (device_name == "multi-cpu") {
			std::cout << "- Number of qubits:" << nqubits << std::endl;
            std::cout << "- Number of global qubits:" << global_nqubits << std::endl;
            std::cout << "- Number of local qubits:" << nqubits - global_nqubits << std::endl;
		}
	}

    QuantumCircuit circuit(nqubits);
    circuit.add_X_gate(0);

    /* auto merged_gate = gate::merge(gate::CNOT(0, 1),gate::Y(1));
    circuit.add_gate(merged_gate);
    circuit.add_RX_gate(3, 0.5); */
    circuit.add_SWAP_gate(0, 3);
    //circuit.update_quantum_state(&ref_state);
    circuit.update_quantum_state(&state);

    // sampling
    //   1st param. is number of sampling.
    //   2nd param. is random-seed.
    // You must call state.sampling with the same random seed in all mpi-ranks
    //std::vector<ITYPE> ref_sample = ref_state.sampling(50, 2021);
    std::vector<ITYPE> sample = state.sampling(50, 2021);
    if (mpirank == 0) {
        std::cout << "Sampling = ";
        for (const auto& e : sample) std::cout << e << " ";
        std::cout << std::endl;

        /* std::cout << "Sampling(single cpu) = ";
        for (const auto& e : ref_sample) std::cout << e << " ";
        std::cout << std::endl; */
    }

    Observable observable(nqubits);
    observable.add_operator(2.0, "X 2 Y 1 Z 0");
    observable.add_operator(-3.0, "Z 2");
    //auto ref_value = observable.get_expectation_value(&ref_state);
    auto value = observable.get_expectation_value(&state);
    if (mpirank == 0)
        std::cout << "Expectation value = " << value << std::endl;
    //std::cout << "Expectation value(single cpu) = " << ref_value << std::endl;

    MPI_Finalize();
    return 0;
}