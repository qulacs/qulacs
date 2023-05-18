#include "optimizer.hpp"

AdamOptimizer::AdamOptimizer(UINT trainable_parameter_count,
    double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999,
    double epsilon = 1e-8)
    : GradientBasedOptimizer(trainable_parameter_count),
      _m(trainable_parameter_count, 0),
      _v(trainable_parameter_count, 0),
      _learning_rate(learning_rate),
      _beta1(beta1),
      _beta2(beta2),
      _epsilon(epsilon),
      _t(1) {}

AdamOptimizer::~AdamOptimizer() {}

void AdamOptimizer::apply_gradient(
    std::vector<double>* parameter, const std::vector<double>& gradient) {
    _learning_rate *= sqrt(1 - pow(_beta2, _t)) / (1 - pow(_beta1, _t));
    for (UINT i = 0; i < (*parameter).size(); ++i) {
        _m[i] = _beta1 * _m[i] + (1 - _beta1) * gradient[i];
        _v[i] = _beta2 * _v[i] + (1 - _beta2) * gradient[i] * gradient[i];
        (*parameter)[i] -= _learning_rate * _m[i] / (sqrt(_v[i]) + _epsilon);
    }
}

GradientDecentOptimizer::GradientDecentOptimizer(
    UINT trainable_parameter_count, double learning_rate = 0.01)
    : GradientBasedOptimizer(trainable_parameter_count),
      _learning_rate(learning_rate) {}

GradientDecentOptimizer::~GradientDecentOptimizer() {}

void GradientDecentOptimizer::apply_gradient(
    std::vector<double>* parameter, const std::vector<double>& gradient) {
    for (UINT i = 0; i < (*parameter).size(); ++i) {
        (*parameter)[i] -= _learning_rate * gradient[i];
    }
}
