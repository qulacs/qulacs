#pragma once

#include <cppsim/type.hpp>
#include <cppsim/state.hpp>
#include <cppsim/observable.hpp>
#include "boolean_formula.hpp"
#include "loss_function.hpp"
#include <functional>
#include <algorithm>
#include <vector>

class ClassificationProblem {
private:
    std::function<double(std::vector<double>, UINT)> _loss_function = loss_function::softmax_cross_entropy_category;
    std::vector<std::vector<double>> _input_data;
    std::vector<UINT> _label_data;
    UINT _category_count;
public:
    ClassificationProblem(std::vector<std::vector<double>> input_data, std::vector<UINT> label_data) {
        _input_data.swap(input_data);
        _label_data.swap(label_data);
        _category_count = (*std::max_element(_label_data.begin(), _label_data.end()));
    }
    virtual UINT get_input_dim() const {
        return (UINT)_input_data[0].size();
    }
    virtual std::vector<double> get_input_data(UINT sample_id) const {
        return _input_data[sample_id];
    }
    virtual UINT get_category_count() const {
        return _category_count;
    }
    virtual UINT get_output_data(UINT sample_id) const {
        return _label_data[sample_id];
    }
    virtual double compute_loss(UINT sample_id, std::vector<double> probability_distribution) const {
        return _loss_function(probability_distribution, _label_data[sample_id]);
    }
};

class RegressionProblem {
protected:
    std::function<double(std::vector<double>, std::vector<double>)> _loss_function = loss_function::L2_distance<double>;
    std::vector<std::vector<double>> _input_data;
    std::vector<std::vector<double>> _output_data;
public:
    RegressionProblem(std::vector<std::vector<double>> input_data, std::vector<std::vector<double>> output_data) {
        _input_data.swap(input_data);
        _output_data.swap(output_data);
    }
    virtual UINT get_input_dim() const {
        return (UINT)_input_data[0].size();
    }
    virtual std::vector<double> get_input_data(UINT sample_id) const {
        return _input_data[sample_id];
    }
    virtual UINT get_output_dim() const {
        return (UINT)_output_data[0].size();
    }
    virtual std::vector<double> get_output_data(UINT sample_id) {
        return _output_data[sample_id];
    }
    virtual double compute_loss(UINT sample_id, std::vector<double> prediction) {
        return _loss_function(prediction, _output_data[sample_id]);
    };
};

class EnergyMinimizationProblem {
private:
    Observable* _observable;
public:
    EnergyMinimizationProblem(Observable* observable) : _observable(observable) {};
    virtual ~EnergyMinimizationProblem() {
        delete _observable;
    }

    virtual UINT get_term_count() const { return _observable->get_term_count(); }
    virtual const PauliOperator* get_Pauli_operator(UINT index) const { return _observable->get_term(index); }
    virtual ITYPE get_state_dim() const { return _observable->get_state_dim(); }
    virtual UINT get_qubit_count() const { return _observable->get_qubit_count(); }
    virtual double compute_loss(const QuantumStateBase* state) const {
        return _observable->get_expectation_value(state).real();
    };
};

class BooleanOptimizationProblem {
private:
    BooleanFormula* _boolean_formula;
    std::vector<UINT> to_binary_string(ITYPE value) const {
        std::vector<UINT> binary_string(_boolean_formula->get_variable_count(), 0);
        for (UINT i = 0; i < binary_string.size(); ++i) {
            binary_string[i] = value % 2;
            value /= 2;
        }
        return binary_string;
    }
public:
    BooleanOptimizationProblem(BooleanFormula* boolean_formula) : _boolean_formula(boolean_formula) {};
    virtual double compute_loss(const std::vector<UINT>& binary_string) const {
        return _boolean_formula->evaluate(binary_string);
    }
    virtual double compute_loss(const std::vector<double> answer_distribution) const {
        double sum = 0;
        for (ITYPE i = 0; i < (ITYPE)answer_distribution.size(); ++i) {
            auto binary_string = this->to_binary_string(i);
            sum += answer_distribution[i] * _boolean_formula->evaluate(binary_string);
        }
        return sum;
    }
};

