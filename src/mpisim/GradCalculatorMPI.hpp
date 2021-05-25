#include "cppsim/observable.hpp"
#include "cppsim/state.hpp"
#include "vqcsim/parametric_circuit.hpp"

class DllExport GradCalculatorMPI {
public:
    std::vector<std::complex<double>> calculate_grad(
        ParametricQuantumCircuit& x, Observable& obs, std::vector<double> theta);
};