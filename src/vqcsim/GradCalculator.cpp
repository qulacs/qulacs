#define _USE_MATH_DEFINES
#include "GradCalculator.hpp"

#include "../cppsim/causal_cone.hpp"

std::vector<std::complex<double>> GradCalculator::calculate_grad(
    ParametricQuantumCircuit& x, Observable& obs, std::vector<double> theta) {
    std::vector<std::complex<double>> ans;
    Causal hoge;
    for (int i = 0; i < x.get_parameter_count(); ++i) {
        std::complex<double> y, z;
        {
            for (int q = 0; q < x.get_parameter_count(); ++q) {
                float diff = 0;
                if (i == q) {
                    diff = M_PI / 2.0;
                }
                x.set_parameter(q, theta[q] + diff);
            }
            y = hoge.CausalCone(x, obs);
        }
        {
            for (int q = 0; q < x.get_parameter_count(); ++q) {
                float diff = 0;
                if (i == q) {
                    diff = M_PI / 2.0;
                }
                x.set_parameter(q, theta[q] - diff);
            }
            z = hoge.CausalCone(x, obs);
        }
        ans.push_back((y - z) / 2.0);
    }
    return ans;
};
