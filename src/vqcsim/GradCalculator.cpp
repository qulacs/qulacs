#include "GradCalculator.hpp"

std::vector<std::complex<double>> GradCalculator::calculate_grad(ParametricQuantumCircuit &x,Observable &obs,double theta){
    std::vector<std::complex<double>> ans;

    for(int i = 0;i < x.get_parameter_count();++i){
        std::complex<double> y,z;
        {
            QuantumState state(x.qubit_count);
            for(int q = 0;q < x.get_parameter_count();++q){
                float diff = 0;
                if(i == q){
                    diff = M_PI / 2.0;
                }
                x.set_parameter(q,theta + diff);
            }
            state.set_zero_state();
            x.update_quantum_state(&state);
            y = obs.get_expectation_value(&state);
        }
        {
            QuantumState state(x.qubit_count);
            for(int q = 0;q < x.get_parameter_count();++q){
                float diff = 0;
                if(i == q){
                    diff = M_PI / 2.0;
                }
                x.set_parameter(q,theta - diff);
            }
            state.set_zero_state();
            x.update_quantum_state(&state);
            z = obs.get_expectation_value(&state);
        }
        ans.push_back((y-z)/2.0);
    }
    return ans;
};