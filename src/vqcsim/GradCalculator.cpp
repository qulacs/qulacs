#define _USE_MATH_DEFINES
#include "GradCalculator.hpp"

#include "causalcone_simulator.hpp"
//クラスのメンバ関数だと使い方がわかりません
std::vector<std::complex<double>> GradCalculator::calculate_grad(
    ParametricQuantumCircuit& x, Observable& obs, std::vector<double> theta) {
    std::vector<std::complex<double>> ans;

    for (UINT i = 0; i < x.get_parameter_count(); ++i) {
        std::complex<double> y, z;
        {
            for (UINT q = 0; q < x.get_parameter_count(); ++q) {
                float diff = 0;
                if (i == q) {
                    diff = M_PI / 2.0;
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
                    diff = M_PI / 2.0;
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

std::vector<std::complex<double>> calculate_grad(
    ParametricQuantumCircuit& x, Observable& obs, std::vector<double> theta) {
    std::vector<std::complex<double>> ans;

    for (UINT i = 0; i < x.get_parameter_count(); ++i) {
        std::complex<double> y, z;
        {
            for (UINT q = 0; q < x.get_parameter_count(); ++q) {
                float diff = 0;
                if (i == q) {
                    diff = M_PI / 2.0;
                }
                x.set_parameter(q, theta[q] + diff);
            }
            QuantumState state(x.qubit_count);
            x.update_quantum_state(&state);
            y =obs.get_expectation_value(&state);
        }
        {
            for (UINT q = 0; q < x.get_parameter_count(); ++q) {
                float diff = 0;
                if (i == q) {
                    diff = M_PI / 2.0;
                }
                x.set_parameter(q, theta[q] - diff);
            }
            QuantumState state(x.qubit_count);
            x.update_quantum_state(&state);
            z =obs.get_expectation_value(&state);
        }
        //std::cerr<<y<<" "<<z<<std::endl;
        ans.push_back((y - z) / 2.0);
    }
    return ans;
};