#include "GradCalculatorMPI.hpp"

#include <mpi.h>

#include "utils.hpp"

std::vector<std::complex<double>> GradCalculatorMPI::calculate_grad(
    ParametricQuantumCircuit& x, Observable& obs, double theta) {
    std::vector<std::complex<double>> node_ans;
    int myrank, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    for (int i = myrank; i < x.get_parameter_count(); i += numprocs) {
        std::complex<double> y, z;
        {
            QuantumState state(x.qubit_count);
            for (int q = 0; q < x.get_parameter_count(); ++q) {
                float diff = 0;
                if (i == q) {
                    diff = M_PI / 2.0;
                }
                x.set_parameter(q, theta + diff);
            }
            state.set_zero_state();
            x.update_quantum_state(&state);
            y = obs.get_expectation_value(&state);
        }
        {
            QuantumState state(x.qubit_count);
            for (int q = 0; q < x.get_parameter_count(); ++q) {
                float diff = 0;
                if (i == q) {
                    diff = M_PI / 2.0;
                }
                x.set_parameter(q, theta - diff);
            }
            state.set_zero_state();
            x.update_quantum_state(&state);
            z = obs.get_expectation_value(&state);
        }
        node_ans.push_back((y - z) / 2.0);
    }
    std::vector<std::complex<double>> ans;
    if (myrank == 0) {
        // gather data.
        std::vector<std::vector<std::complex<double>>> data(numprocs);
        data[0] = node_ans;
        for (int q = 1; q < numprocs; ++q) {
            Utility::receive_vector(0, q, 0, data[q]);
        }
        std::cout << x.get_parameter_count() << std::endl;
        ans.resize(x.get_parameter_count());
        for (int q = 0; q < x.get_parameter_count(); ++q) {
            ans[q] = (data[q % numprocs][q / numprocs]);
        }
    } else {
        Utility::send_vector(myrank, 0, 0, node_ans);
    }
    return ans;
};