#include "cppsim/observable.hpp"
#include "parametric_circuit.hpp"
#include "cppsim/state.hpp"

class DllExport GradCalculator{
    public:
        std::vector<std::complex<double>> calculate_grad(ParametricQuantumCircuit &x,Observable &obs,double theta);
};