
#include "differential.hpp"

double GradientByHalfPi::compute_gradient(ParametricQuantumCircuitSimulator* sim, const EnergyMinimizationProblem* instance, const std::vector<double>& parameter, std::vector<double>* gradient) {
    UINT previous_pos = 0;
    UINT current_pos = 0;

    UINT last_pos = sim->get_gate_count();
    sim->initialize_state();

    for (UINT i = 0; i < parameter.size(); ++i) {
        sim->set_parameter_value(i, parameter[i]);
    }
    for (UINT i = 0; i < parameter.size(); ++i) {
        current_pos = sim->get_parametric_gate_position(i);
        sim->simulate_range(previous_pos, current_pos);

        sim->copy_state_to_buffer();
        sim->add_parameter_value(i, +M_PI / 4);
        sim->simulate_range(current_pos, last_pos);
        double res1 = instance->compute_loss(sim->get_state_ptr());

        sim->copy_state_from_buffer();
        sim->add_parameter_value(i, -M_PI/2);
        sim->simulate_range(current_pos, last_pos);
        double res2 = instance->compute_loss(sim->get_state_ptr());

        (*gradient)[i] = (res1 - res2) / 2;

        sim->copy_state_from_buffer();
        sim->add_parameter_value(i, +M_PI / 4);
        previous_pos = current_pos;
    }
    sim->simulate_range(previous_pos, last_pos);
    double loss = instance->compute_loss(sim->get_state_ptr());
    return loss;
}