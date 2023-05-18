#pragma once

#include <algorithm>
#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>
#include <cppsim/type.hpp>
#include <functional>
#include <vector>

#include "boolean_formula.hpp"
#include "loss_function.hpp"

class ClassificationProblem {
private:
    std::function<double(std::vector<double>, UINT)> _loss_function =
        loss_function::softmax_cross_entropy_category;
    std::vector<std::vector<double>> _input_data;
    std::vector<UINT> _label_data;
    UINT _category_count;

public:
    ClassificationProblem(std::vector<std::vector<double>> input_data,
        std::vector<UINT> label_data);

    virtual UINT get_input_dim() const;

    virtual std::vector<double> get_input_data(UINT sample_id) const;

    virtual UINT get_category_count() const;

    virtual UINT get_output_data(UINT sample_id) const;

    virtual double compute_loss(
        UINT sample_id, std::vector<double> probability_distribution) const;
};

class RegressionProblem {
protected:
    std::function<double(std::vector<double>, std::vector<double>)>
        _loss_function = loss_function::L2_distance<double>;
    std::vector<std::vector<double>> _input_data;
    std::vector<std::vector<double>> _output_data;

public:
    RegressionProblem(std::vector<std::vector<double>> input_data,
        std::vector<std::vector<double>> output_data);

    virtual UINT get_input_dim() const;

    virtual std::vector<double> get_input_data(UINT sample_id) const;

    virtual UINT get_output_dim() const;

    virtual std::vector<double> get_output_data(UINT sample_id);

    virtual double compute_loss(UINT sample_id, std::vector<double> prediction);
};

class EnergyMinimizationProblem {
private:
    Observable* _observable;

public:
    EnergyMinimizationProblem(Observable* observable);

    virtual ~EnergyMinimizationProblem();

    virtual UINT get_term_count() const;

    virtual const PauliOperator* get_Pauli_operator(UINT index) const;

    virtual ITYPE get_state_dim() const;

    virtual UINT get_qubit_count() const;

    virtual double compute_loss(const QuantumStateBase* state) const;
};

class BooleanOptimizationProblem {
private:
    BooleanFormula* _boolean_formula;
    std::vector<UINT> to_binary_string(ITYPE value) const;

public:
    BooleanOptimizationProblem(BooleanFormula* boolean_formula);

    virtual double compute_loss(const std::vector<UINT>& binary_string) const;

    virtual double compute_loss(
        const std::vector<double> answer_distribution) const;
};
