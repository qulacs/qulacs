
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include "hamiltonian.hpp"
#include "pauli_operator.hpp"
#include "type.hpp"
#include "utility.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>





Hamiltonian::Hamiltonian(UINT qubit_count){
    _qubit_count = qubit_count;
}

Hamiltonian::Hamiltonian(std::string filename)
{
    UINT qubit_count = 0;

    std::ifstream ifs;
    ifs.open(filename);

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

    _qubit_count = qubit_count;

    for (UINT i = 0; i < ops.size(); ++i){
        this->add_operator(new PauliOperator(ops[i].c_str(), coefs[i].real()));
    }
}

Hamiltonian::~Hamiltonian(){
    for(auto& term : this->_operator_list){
        delete term;
    }
}

void Hamiltonian::add_operator(PauliOperator* mpt){
    this->_operator_list.push_back(mpt);
}

void Hamiltonian::add_operator(double coef, std::string pauli_string) {
	this->add_operator(new PauliOperator(pauli_string, coef));
}

double Hamiltonian::get_expectation_value(const QuantumStateBase* state) const {
	double sum = 0;
	for (auto pauli : this->_operator_list) {
		sum += pauli->get_expectation_value(state);
	}
	return sum;
}

std::pair<Hamiltonian*, Hamiltonian*> Hamiltonian::get_split_hamiltonian(std::string filename){
    UINT qubit_count = 0;

    std::ifstream ifs;
    ifs.open(filename);

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

    Hamiltonian* ham_diag =  new Hamiltonian(qubit_count);
    Hamiltonian* ham_non_diag =  new Hamiltonian(qubit_count);

    for (UINT i = 0; i < ops.size(); ++i){
        if (ops[i].find("X") != std::string::npos || ops[i].find("Y") != std::string::npos){
            ham_non_diag->add_operator(new PauliOperator(ops[i].c_str(), coefs[i].real()));
        }else{
            ham_diag->add_operator(new PauliOperator(ops[i].c_str(), coefs[i].real()));
        }
    }

    return std::make_pair(ham_diag, ham_non_diag);
}
