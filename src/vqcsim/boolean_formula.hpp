#pragma once


#include <cppsim/type.hpp>
#include <vector>
#include <algorithm>

class BooleanFormula {
private:
    std::vector<std::pair<double, std::vector<UINT>>> _formula;
    UINT _variable_count = 0;
public:
    virtual double evaluate(std::vector<UINT> binary_string) {
        double sum = 0.;
        for (auto term : _formula) {
            double coef = term.first;
            auto variable_index_list = term.second;
            double term_value = coef;
            for (auto index : variable_index_list) {
                term_value *= binary_string[index];
            }
            sum += term_value;
        }
        return sum;
    }
    virtual UINT get_variable_count() const {
        return _variable_count;
    }
};


