
#include "loss_function.hpp"

// loss functions 
namespace loss_function {
    template<typename T>
    double L2_distance(const std::vector<T>& s1, const std::vector<T>& s2) {
        double sum = 0;
        for (UINT i = 0; i < s1.size(); ++i) {
            sum += pow(abs(s1[i] - s2[i]), 2.);
        }
        return sum;
    }

    double cross_entropy(const std::vector<double>& prediction, const std::vector<double>& correct_label) {
        double sum = 0;
        for (UINT i = 0; i < prediction.size(); ++i) {
            sum += -correct_label[i] * log(prediction[i]);
        }
        return sum;
    }

    double softmax_cross_entropy(const std::vector<double>& prediction, const std::vector<double>& correct_label) {
        double denominator = 0;
        for (auto val : prediction) {
            denominator += exp(val);
        }
        double sum = 0;
        for (UINT i = 0; i < prediction.size(); ++i) {
            sum += -correct_label[i] * log(exp(prediction[i]) / denominator);
        }
        return sum;
    }

    double softmax_cross_entropy_category(std::vector<double> prediction, UINT correct_label) {
        double denominator = 0;
        for (auto val : prediction) {
            denominator += exp(val);
        }
        return -log(exp(prediction[correct_label]) / denominator);
    }
}

