
#pragma once
#include <cppsim/type.hpp>
#include <vector>

class ParametricQuantumCircuitModel;


class Optimizer {
protected:
    UINT _trainable_parameter_count;
    Optimizer(UINT trainable_parameter_count) : _trainable_parameter_count(trainable_parameter_count) {};
    virtual ~Optimizer() {};
public:
};

class GradientFreeOptimizer : public Optimizer{
public:
    GradientFreeOptimizer(UINT trainable_parameter_count) : Optimizer(trainable_parameter_count) {};
    virtual ~GradientFreeOptimizer() {};
    virtual void update_parameter(std::vector<double>* next_parameter, double previous_loss) = 0;
};

class GradientBasedOptimizer : public Optimizer{
public:
    GradientBasedOptimizer(UINT trainable_parameter_count) : Optimizer(trainable_parameter_count) {};
    virtual ~GradientBasedOptimizer() {};
    virtual void apply_gradient(std::vector<double>* parameter, const std::vector<double>& gradient) = 0;
};

class AdamOptimizer : public GradientBasedOptimizer{
private:
    double _learning_rate;
    double _beta1;            /*<-- decay rate of first momentum */
    double _beta2;            /*<-- decay rate of second momentum */
    double _epsilon;        /*<-- mergin for avoid zero division */
    double _t;                /*<-- decay order of learning rate */
    std::vector<double> _m;    /*<-- first momentum */
    std::vector<double> _v;    /*<-- second momentum */
public:
    AdamOptimizer(UINT trainable_parameter_count, double learning_rate = 0.001, double beta1=0.9, double beta2=0.999, double epsilon=1e-8 )
    :GradientBasedOptimizer(trainable_parameter_count), _m(trainable_parameter_count,0), _v(trainable_parameter_count,0), 
    _learning_rate(learning_rate), _beta1(beta1), _beta2(beta2), _epsilon(epsilon), _t(1){}
    virtual ~AdamOptimizer() {};

    void apply_gradient(std::vector<double>* parameter, const std::vector<double>& gradient) override{
        _learning_rate *= sqrt(1 - pow(_beta2,_t)) / (1 - pow(_beta1, _t));
        for(UINT i=0 ; i<(*parameter).size() ; ++i){
            _m[i] = _beta1 * _m[i] + (1 - _beta1) * gradient[i];
            _v[i] = _beta2 * _v[i] + (1 - _beta2) * gradient[i] * gradient[i];
            (*parameter)[i] -= _learning_rate * _m[i] / (sqrt(_v[i]) + _epsilon);
        }
    }
};

class GradientDecentOptimizer : public GradientBasedOptimizer {
private:
    double _learning_rate;
public:
    GradientDecentOptimizer(UINT trainable_parameter_count, double learning_rate = 0.01)
    :GradientBasedOptimizer(trainable_parameter_count), _learning_rate(learning_rate){
    }
    virtual ~GradientDecentOptimizer() {};

    void apply_gradient(std::vector<double>* parameter, const std::vector<double>& gradient) override {
        for (UINT i = 0; i < (*parameter).size(); ++i) {
            (*parameter)[i] -= _learning_rate * gradient[i];
        }
    }
};
