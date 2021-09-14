#define _USE_MATH_DEFINES
#include "GradCalculator.hpp"

#include "causalcone_simulator.hpp"

std::vector<std::complex<double>> GradCalculator::calculate_grad(
    ParametricQuantumCircuit& x, Observable& obs, std::vector<double> theta) {
    std::vector<std::complex<double>> ans;

    for (UINT i = 0; i < x.get_parameter_count(); ++i) {
        std::complex<double> y, z;
        {
            for (UINT q = 0; q < x.get_parameter_count(); ++q) {
                float diff = 0;
                if (i == q) {
                    diff = 3.1415926589793238462643 / 2.0;
                }
                x.set_parameter(q, theta[q] + diff);
            }
            CausalConeSimulator hoge(x, obs);
            y = hoge.get_expectation_value();
        }
        {
            for (UINT q = 0; q < x.get_parameter_count(); ++q) {
                float diff = 0;
                if (i == q) {
                    diff = 3.1415926589793238462643 / 2.0;
                }
                x.set_parameter(q, theta[q] - diff);
            }
            CausalConeSimulator hoge(x, obs);
            z = hoge.get_expectation_value();
        }
        ans.push_back((y - z) / 2.0);
    }
    return ans;
};

