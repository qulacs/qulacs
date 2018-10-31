
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include "observable.hpp"
#include "pauli_operator.hpp"
#include "type.hpp"
#include "utility.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>


Observable::Observable(UINT qubit_count){
    _qubit_count = qubit_count;
}

Observable::~Observable(){
    for(auto& term : this->_operator_list){
        delete term;
    }
}

void Observable::add_operator(const PauliOperator* mpt){
    PauliOperator* _mpt = mpt->copy();
    this->_operator_list.push_back(_mpt);
}

void Observable::add_operator(double coef, std::string pauli_string) {
    this->add_operator(new PauliOperator(pauli_string, coef));
}

double Observable::get_expectation_value(const QuantumStateBase* state) const {
    double sum = 0;
    for (auto pauli : this->_operator_list) {
        sum += pauli->get_expectation_value(state);
    }
    return sum;
}

CPPCTYPE Observable::get_transition_amplitude(const QuantumStateBase* state_bra, const QuantumStateBase* state_ket) const {
    CPPCTYPE sum = 0;
    for (auto pauli : this->_operator_list) {
        sum += pauli->get_transition_amplitude(state_bra, state_ket);
    }
    return sum;
}

namespace observable{
    Observable* create_observable_from_openfermion_file(std::string file_path){
        UINT qubit_count = 0;

        std::ifstream ifs;
        ifs.open(file_path);

        if (!ifs){
            std::cerr << "ERROR: Cannot open file" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::string str;
        std::vector<CPPCTYPE> coefs;
        std::vector<std::string> ops;
        while (getline(ifs, str)) {
            std::vector<std::string> elems;
            std::vector<std::string> index_list;
            elems = split(str, "()j[]+");

            chfmt(elems[3]);

            CPPCTYPE coef(std::stod(elems[0]), std::stod(elems[1]));
            coefs.push_back(coef);
            ops.push_back(elems[3]);

            index_list = split(elems[3], "XYZ ");
            for (UINT i = 0; i < index_list.size(); ++i){
                UINT n = std::stoi(index_list[i]) + 1;
                if (qubit_count < n)
                    qubit_count = n;
            }
        }
        if (!ifs.eof()){
            std::cerr << "ERROR: Invalid format" << std::endl;
        }
        ifs.close();

        Observable* observable = new Observable(qubit_count);

        for (UINT i = 0; i < ops.size(); ++i){
            observable->add_operator(new PauliOperator(ops[i].c_str(), coefs[i].real()));
        }

        return observable;
    }

    Observable* create_observable_from_openfermion_text(std::string text){
        UINT qubit_count = 0;

        std::vector<std::string> lines;
        std::vector<CPPCTYPE> coefs;
        std::vector<std::string> ops;
        lines = split(text, "\n");

        for (std::string line: lines){
            std::vector<std::string> elems;
            std::vector<std::string> index_list;
            elems = split(line, "()j[]+");

            chfmt(elems[3]);

            CPPCTYPE coef(std::stod(elems[0]), std::stod(elems[1]));
            coefs.push_back(coef);
            ops.push_back(elems[3]);

            index_list = split(elems[3], "XYZ ");
            for (UINT i = 0; i < index_list.size(); ++i){
                UINT n = std::stoi(index_list[i]) + 1;
                if (qubit_count < n)
                    qubit_count = n;
            }
        }

        Observable* observable = new Observable(qubit_count);

        for (UINT i = 0; i < ops.size(); ++i){
            observable->add_operator(new PauliOperator(ops[i].c_str(), coefs[i].real()));
        }

        return observable;
    }

    std::pair<Observable*, Observable*> create_split_observable(std::string file_path){
        UINT qubit_count = 0;

        std::ifstream ifs;
        ifs.open(file_path);

        if (!ifs){
            std::cerr << "ERROR: Cannot open file" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // loading lines and check qubit_count
        std::string str;
        std::vector<CPPCTYPE> coefs;
        std::vector<std::string> ops;

        while (getline(ifs, str)) {
            std::vector<std::string> elems;
            std::vector<std::string> index_list;
            elems = split(str, "()j[]+");

            chfmt(elems[3]);

            CPPCTYPE coef(std::stod(elems[0]), std::stod(elems[1]));
            coefs.push_back(coef);
            ops.push_back(elems[3]);

            index_list = split(elems[3], "XYZ ");
            for (UINT i = 0; i < index_list.size(); ++i){
                UINT n = std::stoi(index_list[i]) + 1;
                if (qubit_count < n)
                    qubit_count = n;
            }
        }
        if (!ifs.eof()){
            std::cerr << "ERROR: Invalid format" << std::endl;
        }
        ifs.close();

        Observable* observable_diag =  new Observable(qubit_count);
        Observable* observable_non_diag =  new Observable(qubit_count);

        for (UINT i = 0; i < ops.size(); ++i){
            if (ops[i].find("X") != std::string::npos || ops[i].find("Y") != std::string::npos){
                observable_non_diag->add_operator(new PauliOperator(ops[i].c_str(), coefs[i].real()));
            }else{
                observable_diag->add_operator(new PauliOperator(ops[i].c_str(), coefs[i].real()));
            }
        }

        return std::make_pair(observable_diag, observable_non_diag);
    }
}


