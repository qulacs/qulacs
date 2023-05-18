#include "problem.hpp"

ClassificationProblem::ClassificationProblem(
    std::vector<std::vector<double>> input_data, std::vector<UINT> label_data) {
    _input_data.swap(input_data);
    _label_data.swap(label_data);
    _category_count =
        (*std::max_element(_label_data.begin(), _label_data.end()));
}

UINT ClassificationProblem::get_input_dim() const {
    return (UINT)_input_data[0].size();
}

std::vector<double> ClassificationProblem::get_input_data(
    UINT sample_id) const {
    return _input_data[sample_id];
}

UINT ClassificationProblem::get_category_count() const {
    return _category_count;
}

UINT ClassificationProblem::get_output_data(UINT sample_id) const {
    return _label_data[sample_id];
}

double ClassificationProblem::compute_loss(
    UINT sample_id, std::vector<double> probability_distribution) const {
    return _loss_function(probability_distribution, _label_data[sample_id]);
}

RegressionProblem::RegressionProblem(
    std::vector<std::vector<double>> input_data,
    std::vector<std::vector<double>> output_data) {
    _input_data.swap(input_data);
    _output_data.swap(output_data);
}

UINT RegressionProblem::get_input_dim() const {
    return (UINT)_input_data[0].size();
}

std::vector<double> RegressionProblem::get_input_data(UINT sample_id) const {
    return _input_data[sample_id];
}

UINT RegressionProblem::get_output_dim() const {
    return (UINT)_output_data[0].size();
}

std::vector<double> RegressionProblem::get_output_data(UINT sample_id) {
    return _output_data[sample_id];
}

double RegressionProblem::compute_loss(
    UINT sample_id, std::vector<double> prediction) {
    return _loss_function(prediction, _output_data[sample_id]);
}

EnergyMinimizationProblem::EnergyMinimizationProblem(Observable* observable)
    : _observable(observable){};

EnergyMinimizationProblem::~EnergyMinimizationProblem() { delete _observable; }

UINT EnergyMinimizationProblem::get_term_count() const {
    return _observable->get_term_count();
}

const PauliOperator* EnergyMinimizationProblem::get_Pauli_operator(
    UINT index) const {
    return _observable->get_term(index);
}

ITYPE EnergyMinimizationProblem::get_state_dim() const {
    return _observable->get_state_dim();
}

UINT EnergyMinimizationProblem::get_qubit_count() const {
    return _observable->get_qubit_count();
}

double EnergyMinimizationProblem::compute_loss(
    const QuantumStateBase* state) const {
    return _observable->get_expectation_value(state).real();
}

BooleanOptimizationProblem::BooleanOptimizationProblem(
    BooleanFormula* boolean_formula)
    : _boolean_formula(boolean_formula){};

std::vector<UINT> BooleanOptimizationProblem::to_binary_string(
    ITYPE value) const {
    std::vector<UINT> binary_string(_boolean_formula->get_variable_count(), 0);
    for (UINT i = 0; i < binary_string.size(); ++i) {
        binary_string[i] = value % 2;
        value /= 2;
    }
    return binary_string;
}

double BooleanOptimizationProblem::compute_loss(
    const std::vector<UINT>& binary_string) const {
    return _boolean_formula->evaluate(binary_string);
}

double BooleanOptimizationProblem::compute_loss(
    const std::vector<double> answer_distribution) const {
    double sum = 0;
    for (ITYPE i = 0; i < (ITYPE)answer_distribution.size(); ++i) {
        auto binary_string = this->to_binary_string(i);
        sum +=
            answer_distribution[i] * _boolean_formula->evaluate(binary_string);
    }
    return sum;
}
