
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include "observable.hpp"
#include "pauli_operator.hpp"
#include "type.hpp"
#include "utility.hpp"
#include "state.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>

void HermitianQuantumOperator::add_operator(const PauliOperator* mpt){
    if (std::abs(mpt->get_coef().imag()) > 0){
        std::cerr << "Error: HermitianQuantumOperator::add_operator(const PauliOperator* mpt): PauliOperator must be Hermitian." << std::endl;
        return;
    }
    GeneralQuantumOperator::add_operator(mpt);
}

void HermitianQuantumOperator::add_operator(CPPCTYPE coef, std::string pauli_string) {
    if (std::abs(coef.imag()) > 0){
        std::cerr << "Error: HermitianQuantumOperator::add_operator(const PauliOperator* mpt): PauliOperator must be Hermitian." << std::endl;
        return;
    }
	GeneralQuantumOperator::add_operator(coef, pauli_string);
}

CPPCTYPE HermitianQuantumOperator::get_expectation_value(const QuantumStateBase* state) const {
    return GeneralQuantumOperator::get_expectation_value(state).real();
}

namespace observable{
    HermitianQuantumOperator* create_observable_from_openfermion_file(std::string file_path){
        UINT qubit_count = 0;

        std::ifstream ifs;
        ifs.open(file_path);
        double coef_real, coef_imag;
        if (!ifs){
            std::cerr << "ERROR: Cannot open file" << std::endl;
			return NULL;
		}

        std::string line;
        std::vector<CPPCTYPE> coefs;
        std::vector<std::string> ops;
        char buf[256];
        char symbol_j[1];
        UINT matches;
        while (getline(ifs, line)) {
            std::vector<std::string> index_list;

            if(line[0]=='('){
                matches = std::sscanf(line.c_str(), "(%lf+%lfj) [%[^]]]", &coef_real, &coef_imag, buf);
                if (matches < 2){
                    matches = std::sscanf(line.c_str(), "(%lf-%lfj) [%[^]]]", &coef_real, &coef_imag, buf);
                    coef_imag = -coef_imag;
                }
                if (matches < 3){
                    std::strcpy(buf, "I0");
                }
            }else{
                matches = std::sscanf(line.c_str(), "%lf%[j] [%[^]]]", &coef_imag, symbol_j, buf);
                coef_real = 0.;
                if (matches < 3){
                    std::strcpy(buf, "I0");
                }
                if (symbol_j[0] != 'j'){
                    matches = std::sscanf(line.c_str(), "%lf [%[^]]]", &coef_real, buf);
                    coef_imag = 0.;
                    if (matches < 2){
                        std::strcpy(buf, "I0");
                    }
                }
                if (matches == 0){
                    continue;
                }
            }

            std::string str_buf(buf, std::strlen(buf));
            chfmt(str_buf);
            CPPCTYPE coef(coef_real, coef_imag);
            coefs.push_back(coef);
            ops.push_back(str_buf);
            index_list = split(str_buf, "IXYZ ");

            for (UINT i = 0; i < index_list.size(); ++i){
                UINT n = std::stoi(index_list[i]) + 1;
                if (qubit_count < n)
                    qubit_count = n;
            }
        }
        if (!ifs.eof()){
            std::cerr << "ERROR: Invalid format" << std::endl;
			return NULL;
		}
        ifs.close();

        HermitianQuantumOperator* observable = new HermitianQuantumOperator(qubit_count);

        for (UINT i = 0; i < ops.size(); ++i){
            observable->add_operator(new PauliOperator(ops[i].c_str(), coefs[i]));
        }

        return observable;
    }

    HermitianQuantumOperator* create_observable_from_openfermion_text(std::string text){
        UINT qubit_count = 0;

        std::vector<std::string> lines;
        std::vector<CPPCTYPE> coefs;
        std::vector<std::string> ops;
        double coef_real, coef_imag;

        lines = split(text, "\n");
        char buf[256];
        char symbol_j[1];
        UINT matches;
        for (std::string line: lines){
            std::vector<std::string> index_list;

            if(line[0]=='('){
                matches = std::sscanf(line.c_str(), "(%lf+%lfj) [%[^]]]", &coef_real, &coef_imag, buf);
                if (matches < 2){
                    matches = std::sscanf(line.c_str(), "(%lf-%lfj) [%[^]]]", &coef_real, &coef_imag, buf);
                    coef_imag = -coef_imag;
                }
                if (matches < 3){
                    std::strcpy(buf, "I0");
                }
            }else{
                matches = std::sscanf(line.c_str(), "%lf%[j] [%[^]]]", &coef_imag, symbol_j, buf);
                coef_real = 0.;
                if (matches < 3){
                    std::strcpy(buf, "I0");
                }
                if (symbol_j[0] != 'j'){
                    matches = std::sscanf(line.c_str(), "%lf [%[^]]]", &coef_real, buf);
                    coef_imag = 0.;
                    if (matches < 2){
                        std::strcpy(buf, "I0");
                    }
                }
                if (matches == 0){
                    continue;
                }
            }

            std::string str_buf(buf, std::strlen(buf));
            chfmt(str_buf);
            CPPCTYPE coef(coef_real, coef_imag);
            coefs.push_back(coef);
            ops.push_back(str_buf);
            index_list = split(str_buf, "IXYZ ");

            for (UINT i = 0; i < index_list.size(); ++i){
                UINT n = std::stoi(index_list[i]) + 1;
                if (qubit_count < n)
                    qubit_count = n;
            }
        }
        HermitianQuantumOperator* hermitian_quantum_operator = new HermitianQuantumOperator(qubit_count);

        for (UINT i = 0; i < ops.size(); ++i){
            hermitian_quantum_operator->add_operator(new PauliOperator(ops[i].c_str(), coefs[i]));
        }

        return hermitian_quantum_operator;

    }

    std::pair<HermitianQuantumOperator*, HermitianQuantumOperator*> create_split_observable(std::string file_path){
        UINT qubit_count = 0;
        double coef_real, coef_imag;
        std::ifstream ifs;
        ifs.open(file_path);

        if (!ifs){
            std::cerr << "ERROR: Cannot open file" << std::endl;
			return std::make_pair((HermitianQuantumOperator*)NULL, (HermitianQuantumOperator*)NULL);
        }

        // loading lines and check qubit_count
        std::string line;
        std::vector<CPPCTYPE> coefs;
        std::vector<std::string> ops;
        char buf[256];
        char symbol_j[1];
        UINT matches;
        while (getline(ifs, line)) {
            std::vector<std::string> index_list;

            if(line[0]=='('){
                matches = std::sscanf(line.c_str(), "(%lf+%lfj) [%[^]]]", &coef_real, &coef_imag, buf);
                if (matches < 2){
                    matches = std::sscanf(line.c_str(), "(%lf-%lfj) [%[^]]]", &coef_real, &coef_imag, buf);
                    coef_imag = -coef_imag;
                }
                if (matches < 3){
                    std::strcpy(buf, "I0");
                }
            }else{
                matches = std::sscanf(line.c_str(), "%lf%[j] [%[^]]]", &coef_imag, symbol_j, buf);
                coef_real = 0.;
                if (matches < 3){
                    std::strcpy(buf, "I0");
                }
                if (symbol_j[0] != 'j'){
                    matches = std::sscanf(line.c_str(), "%lf [%[^]]]", &coef_real, buf);
                    coef_imag = 0.;
                    if (matches < 2){
                        std::strcpy(buf, "I0");
                    }
                }
                if (matches == 0){
                    continue;
                }
            }

            std::string str_buf(buf, std::strlen(buf));
            chfmt(str_buf);
            CPPCTYPE coef(coef_real, coef_imag);
            coefs.push_back(coef);
            ops.push_back(str_buf);
            index_list = split(str_buf, "IXYZ ");

            for (UINT i = 0; i < index_list.size(); ++i){
                UINT n = std::stoi(index_list[i]) + 1;
                if (qubit_count < n)
                    qubit_count = n;
            }
        }
        if (!ifs.eof()){
            std::cerr << "ERROR: Invalid format" << std::endl;
			return std::make_pair((HermitianQuantumOperator*)NULL, (HermitianQuantumOperator*)NULL);
		}
        ifs.close();

        HermitianQuantumOperator* observable_diag =  new HermitianQuantumOperator(qubit_count);
        HermitianQuantumOperator* observable_non_diag =  new HermitianQuantumOperator(qubit_count);

        for (UINT i = 0; i < ops.size(); ++i){
            if (ops[i].find("X") != std::string::npos || ops[i].find("Y") != std::string::npos){
                observable_non_diag->add_operator(new PauliOperator(ops[i].c_str(), coefs[i]));
            }else{
                observable_diag->add_operator(new PauliOperator(ops[i].c_str(), coefs[i]));
            }
        }

        return std::make_pair(observable_diag, observable_non_diag);
    }
}


