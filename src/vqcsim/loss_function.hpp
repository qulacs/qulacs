#pragma once

#include <cppsim/type.hpp>
#include <vector>

// loss functions 
namespace loss_function {
    template<typename T>
    double L2_distance(const std::vector<T>& s1, const std::vector<T>& s2);
    double cross_entropy(const std::vector<double>& prediction, const std::vector<double>& correct_label);
    double softmax_cross_entropy(const std::vector<double>& prediction, const std::vector<double>& correct_label);
    double softmax_cross_entropy_category(std::vector<double> prediction, UINT correct_label);

}

