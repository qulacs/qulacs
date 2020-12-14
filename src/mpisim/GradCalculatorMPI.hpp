#include "cppsim/observable.hpp"
#include "vqcsim/parametric_circuit.hpp"
#include "cppsim/state.hpp"

class DllExport GradCalculatorMPI{
    public:
        std::vector<std::complex<double>> calculate_grad(ParametricQuantumCircuit &x,Observable &obs,double theta);
};