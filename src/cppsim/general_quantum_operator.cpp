#include <cstring>

#include "type.hpp"
#include "utility.hpp"
#include <fstream>

#ifndef _MSC_VER
extern "C"{
#include <csim/stat_ops.h>
}
#else
#include <csim/stat_ops.h>
#endif
#include "general_quantum_operator.hpp"
#include "pauli_operator.hpp"
#include "state.hpp"

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
        std::vector<CPPCTYPE> coefs;
        std::vector<std::string> ops;

        // loading lines and check qubit_count
        double coef_real, coef_imag;
        std::string str_buf;
        std::vector<std::string> index_list;

        std::ifstream ifs;
        std::string line;
        ifs.open(file_path);

        while (getline(ifs, line)) {

            std::tuple<double, double, std::string> parsed_items = parse_openfermion_line(line);
            coef_real = std::get<0>(parsed_items);
            coef_imag = std::get<1>(parsed_items);
            str_buf = std::get<2>(parsed_items);

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
			return (GeneralQuantumOperator*)NULL;
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
        std::vector<CPPCTYPE> coefs;
        std::vector<std::string> ops;

        double coef_real, coef_imag;
        std::string str_buf;
        std::vector<std::string> index_list;

        std::vector<std::string> lines;
        lines = split(text, "\n");
        for (std::string line: lines){

            std::tuple<double, double, std::string> parsed_items = parse_openfermion_line(line);
            coef_real = std::get<0>(parsed_items);
            coef_imag = std::get<1>(parsed_items);
            str_buf = std::get<2>(parsed_items);

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
        std::vector<CPPCTYPE> coefs;
        std::vector<std::string> ops;

        std::ifstream ifs;
        ifs.open(file_path);

        if (!ifs){
            std::cerr << "ERROR: Cannot open file" << std::endl;
			return std::make_pair((GeneralQuantumOperator*)NULL, (GeneralQuantumOperator*)NULL);
		}

        // loading lines and check qubit_count
        double coef_real, coef_imag;
        std::string str_buf;
        std::vector<std::string> index_list;

        std::string line;
        while (getline(ifs, line)) {

            std::tuple<double, double, std::string> parsed_items = parse_openfermion_line(line);
            coef_real = std::get<0>(parsed_items);
            coef_imag = std::get<1>(parsed_items);
            str_buf = std::get<2>(parsed_items);
            if (str_buf == (std::string) NULL){
                continue;
            }
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



bool check_Pauli_operator(const GeneralQuantumOperator* quantum_operator, const PauliOperator* pauli_operator) {
	auto vec = pauli_operator->get_index_list();
	UINT val = 0;
	if (vec.size() > 0) {
		val = std::max(val, *std::max_element(vec.begin(), vec.end()));
	}
	return val < (quantum_operator->get_qubit_count());
}
