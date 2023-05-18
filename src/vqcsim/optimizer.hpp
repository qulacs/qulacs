
#pragma once
#include <cppsim/type.hpp>
#include <vector>

class ParametricQuantumCircuitModel;

class Optimizer {
protected:
    UINT _trainable_parameter_count;
    Optimizer(UINT trainable_parameter_count)
        : _trainable_parameter_count(trainable_parameter_count){};
    virtual ~Optimizer(){};

public:
};

class GradientFreeOptimizer : public Optimizer {
public:
    GradientFreeOptimizer(UINT trainable_parameter_count)
        : Optimizer(trainable_parameter_count){};
    virtual ~GradientFreeOptimizer(){};
    virtual void update_parameter(
        std::vector<double>* next_parameter, double previous_loss) = 0;
};

class GradientBasedOptimizer : public Optimizer {
public:
    GradientBasedOptimizer(UINT trainable_parameter_count)
        : Optimizer(trainable_parameter_count){};
    virtual ~GradientBasedOptimizer(){};
    virtual void apply_gradient(std::vector<double>* parameter,
        const std::vector<double>& gradient) = 0;
};

class AdamOptimizer : public GradientBasedOptimizer {
private:
    double _learning_rate;
    double _beta1;          /*<-- decay rate of first momentum */
    double _beta2;          /*<-- decay rate of second momentum */
    double _epsilon;        /*<-- mergin for avoid zero division */
    double _t;              /*<-- decay order of learning rate */
    std::vector<double> _m; /*<-- first momentum */
    std::vector<double> _v; /*<-- second momentum */
public:
    AdamOptimizer(UINT trainable_parameter_count, double learning_rate = 0.001,
        double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
    virtual ~AdamOptimizer();

    void apply_gradient(std::vector<double>* parameter,
        const std::vector<double>& gradient) override;
};

class GradientDecentOptimizer : public GradientBasedOptimizer {
private:
    double _learning_rate;

public:
    GradientDecentOptimizer(
        UINT trainable_parameter_count, double learning_rate = 0.01);

    virtual ~GradientDecentOptimizer();

    void apply_gradient(std::vector<double>* parameter,
        const std::vector<double>& gradient) override;
};
