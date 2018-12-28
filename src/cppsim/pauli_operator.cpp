#include <cassert>
#include <cstring>

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include "type.hpp"
#include "utility.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>


#ifndef _MSC_VER
extern "C"{
#include <csim/stat_ops.h>
}
#else
#include <csim/stat_ops.h>
#endif
#include "pauli_operator.hpp"
#include "state.hpp"


PauliOperator::PauliOperator(std::string strings, CPPCTYPE coef){
    _coef = coef;
    std::stringstream ss(strings);
    std::string pauli_str;
    UINT index, pauli_type=0;
    while(!ss.eof()){
        ss >> pauli_str >> index;
        if (pauli_str.length() == 0) break;
        if(pauli_str=="I" || pauli_str=="i") pauli_type = 0;
        else if(pauli_str=="X" || pauli_str=="x") pauli_type = 1;
        else if(pauli_str=="Y" || pauli_str=="y") pauli_type = 2;
        else if(pauli_str=="Z" || pauli_str=="z") pauli_type = 3;
        else {
            fprintf(stderr, "invalid Pauli string is given : %s\n ",pauli_str.c_str());
            assert(false);
        }
        if(pauli_type!=0) this->add_single_Pauli(index,pauli_type);
    }
}

PauliOperator::PauliOperator(const std::vector<UINT>& target_qubit_list, std::string Pauli_operator_type_list, CPPCTYPE coef){
    _coef = coef;
    UINT term_count = (UINT)(strlen(Pauli_operator_type_list.c_str()));
    UINT pauli_type = 0;
    for(UINT term_index=0;term_index<term_count;++term_index){
        if(Pauli_operator_type_list[term_index] == 'i' || Pauli_operator_type_list[term_index] == 'I'){
            pauli_type = 0;
        }else if(Pauli_operator_type_list[term_index] == 'x' || Pauli_operator_type_list[term_index] == 'X'){
            pauli_type = 1;
        }else if(Pauli_operator_type_list[term_index] == 'y' || Pauli_operator_type_list[term_index] == 'Y'){
            pauli_type = 2;
        }else if(Pauli_operator_type_list[term_index] == 'z' || Pauli_operator_type_list[term_index] == 'Z'){
            pauli_type = 3;
        }else{
            fprintf(stderr, "invalid Pauli string is given\n");
            assert(false);
        }

        if(pauli_type!=0) this->add_single_Pauli(target_qubit_list[term_count],pauli_type);
    }
}

PauliOperator::PauliOperator(const std::vector<UINT>& pauli_list, CPPCTYPE coef) {
    _coef = coef;
    for (UINT term_index = 0; term_index < pauli_list.size(); ++term_index) {
        if (pauli_list[term_index] != 0)
            this->add_single_Pauli(term_index, pauli_list[term_index]);
    }
}

PauliOperator::PauliOperator(const std::vector<UINT>& target_qubit_index_list, const std::vector<UINT>& target_qubit_pauli_list, CPPCTYPE coef) {
    _coef = coef;
    assert(target_qubit_index_list.size() == target_qubit_pauli_list.size());
    for (UINT term_index = 0; term_index < target_qubit_index_list.size(); ++term_index) {
        this->add_single_Pauli(target_qubit_index_list[term_index], target_qubit_pauli_list[term_index]);
    }
}

void PauliOperator::add_single_Pauli(UINT qubit_index, UINT pauli_type){
    this->_pauli_list.push_back(SinglePauliOperator(qubit_index, pauli_type));
}

CPPCTYPE PauliOperator::get_expectation_value(const QuantumStateBase* state) const {
    return _coef * expectation_value_multi_qubit_Pauli_operator_partial_list(
        this->get_index_list().data(),
        this->get_pauli_id_list().data(),
        (UINT)this->get_index_list().size(),
        state->data_c(),
        state->dim
    );
}

CPPCTYPE PauliOperator::get_transition_amplitude(const QuantumStateBase* state_bra, const QuantumStateBase* state_ket) const {
    return _coef * (CPPCTYPE)transition_amplitude_multi_qubit_Pauli_operator_partial_list(
        this->get_index_list().data(),
        this->get_pauli_id_list().data(),
        (UINT)this->get_index_list().size(),
        state_bra->data_c(),
        state_ket->data_c(),
        state_bra->dim
    );
}


PauliOperator* PauliOperator::copy() const {
    auto pauli = new PauliOperator(this->_coef);
    for (auto val : this->_pauli_list) {
        pauli->add_single_Pauli(val.index(), val.pauli_id());
    }
    return pauli;
}


bool check_Pauli_operator(const GeneralQuantumOperator* observable, const PauliOperator* pauli_operator) {
	auto vec = pauli_operator->get_index_list();
	UINT val = 0;
	if (vec.size() > 0) {
		val = std::max(val, *std::max_element(vec.begin(), vec.end()));
	}
	return val < (observable->get_qubit_count());
}

GeneralQuantumOperator::GeneralQuantumOperator(UINT qubit_count){
    _qubit_count = qubit_count;
    _is_hermitian = true;
}

GeneralQuantumOperator::~GeneralQuantumOperator(){
    for(auto& term : this->_operator_list){
        delete term;
    }
}

void GeneralQuantumOperator::add_operator(const PauliOperator* mpt){
    PauliOperator* _mpt = mpt->copy();
	if (!check_Pauli_operator(this, _mpt)) {
		std::cerr << "Error: GeneralQuantumOperator::add_operator(const PauliOperator*): pauli_operator applies target qubit of which the index is larger than qubit_count" << std::endl;
		return;
	}
    if (this->_is_hermitian && std::abs(_mpt->get_coef().imag()) > 0){
        this->_is_hermitian = false;
    }
    this->_operator_list.push_back(_mpt);
}

void GeneralQuantumOperator::add_operator(CPPCTYPE coef, std::string pauli_string) {
	PauliOperator* _mpt = new PauliOperator(pauli_string, coef);
	if (!check_Pauli_operator(this, _mpt)) {
		std::cerr << "Error: GeneralQuantumOperator::add_operator(double,std::string): pauli_operator applies target qubit of which the index is larger than qubit_count" << std::endl;
		return;
	}
    if (this->_is_hermitian && std::abs(coef.imag()) > 0){
        this->_is_hermitian = false;
    }
	this->add_operator(_mpt);
}

CPPCTYPE GeneralQuantumOperator::get_expectation_value(const QuantumStateBase* state) const {
	if (this->_qubit_count != state->qubit_count) {
		std::cerr << "Error: GeneralQuantumOperator::get_expectation_value(const QuantumStateBase*): invalid qubit count" << std::endl;
		return 0.;
	}
    CPPCTYPE sum = 0;
    for (auto pauli : this->_operator_list) {
        sum += pauli->get_expectation_value(state);
    }
    return sum;
}

CPPCTYPE GeneralQuantumOperator::get_transition_amplitude(const QuantumStateBase* state_bra, const QuantumStateBase* state_ket) const {
	if (this->_qubit_count != state_bra->qubit_count || this->_qubit_count != state_ket->qubit_count) {
		std::cerr << "Error: GeneralQuantumOperator::get_transition_amplitude(const QuantumStateBase*, const QuantumStateBase*): invalid qubit count" << std::endl;
		return 0.;
	}

	CPPCTYPE sum = 0;
    for (auto pauli : this->_operator_list) {
        sum += pauli->get_transition_amplitude(state_bra, state_ket);
    }
    return sum;
}

namespace quantum_operator{
    GeneralQuantumOperator* create_general_quantum_operator_from_openfermion_file(std::string file_path){
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

        GeneralQuantumOperator* general_quantum_operator = new GeneralQuantumOperator(qubit_count);

        for (UINT i = 0; i < ops.size(); ++i){
            general_quantum_operator->add_operator(new PauliOperator(ops[i].c_str(), coefs[i]));
        }

        return general_quantum_operator;
    }

    GeneralQuantumOperator* create_general_quantum_operator_from_openfermion_text(std::string text){
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
        GeneralQuantumOperator* general_quantum_operator = new GeneralQuantumOperator(qubit_count);

        for (UINT i = 0; i < ops.size(); ++i){
            general_quantum_operator->add_operator(new PauliOperator(ops[i].c_str(), coefs[i]));
        }

        return general_quantum_operator;
    }

    std::pair<GeneralQuantumOperator*, GeneralQuantumOperator*> create_split_general_quantum_operator(std::string file_path){
        UINT qubit_count = 0;
        UINT imag_idx;
        UINT str_idx;
        std::ifstream ifs;
        ifs.open(file_path);

        if (!ifs){
            std::cerr << "ERROR: Cannot open file" << std::endl;
			return std::make_pair((GeneralQuantumOperator*)NULL, (GeneralQuantumOperator*)NULL);
        }

        // loading lines and check qubit_count
        std::string line;
        std::vector<CPPCTYPE> coefs;
        std::vector<std::string> ops;
        double coef_real, coef_imag;

        while (getline(ifs, line)) {
            std::vector<std::string> elems;
            std::vector<std::string> index_list;
            elems = split(line, "()[]+");
            if (elems.size() < 3){
                continue;
            }

            imag_idx = 1;
            str_idx = 3;
            if (elems[0].find("j") != std::string::npos){
                coef_real = 0;
                imag_idx = 0;
                str_idx = 1;
            } else if (elems[1].find("j") == std::string::npos){
                coef_real = std::stod(elems[imag_idx-1]);
            } else {
                continue;
            }

            coef_imag = std::stod(elems[imag_idx]);
            chfmt(elems[str_idx]);

            CPPCTYPE coef(coef_real, coef_imag);
            coefs.push_back(coef);
            ops.push_back(elems[str_idx]);

            index_list = split(elems[str_idx], "XYZ ");
            for (UINT i = 0; i < index_list.size(); ++i){
                UINT n = std::stoi(index_list[i]) + 1;
                if (qubit_count < n)
                    qubit_count = n;
            }
        }
        if (!ifs.eof()){
            std::cerr << "ERROR: Invalid format" << std::endl;
			return std::make_pair((GeneralQuantumOperator*)NULL, (GeneralQuantumOperator*)NULL);
		}
        ifs.close();

        GeneralQuantumOperator* general_quantum_operator_diag =  new GeneralQuantumOperator(qubit_count);
        GeneralQuantumOperator* general_quantum_operator_non_diag =  new GeneralQuantumOperator(qubit_count);

        for (UINT i = 0; i < ops.size(); ++i){
            if (ops[i].find("X") != std::string::npos || ops[i].find("Y") != std::string::npos){
                general_quantum_operator_non_diag->add_operator(new PauliOperator(ops[i].c_str(), coefs[i]));
            }else{
                general_quantum_operator_diag->add_operator(new PauliOperator(ops[i].c_str(), coefs[i]));
            }
        }

        return std::make_pair(general_quantum_operator_diag, general_quantum_operator_non_diag);
    }
}


