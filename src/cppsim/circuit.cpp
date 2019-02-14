#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstring>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include "gate.hpp"
#include "gate_matrix.hpp"
#include "gate_factory.hpp"
#include "circuit.hpp"
#include "gate_factory.hpp"
#include "pauli_operator.hpp"
#include "observable.hpp"

bool check_gate_index(const QuantumCircuit* circuit, const QuantumGateBase* gate);

void QuantumCircuit::update_quantum_state(QuantumStateBase* state){
	if (state->qubit_count != this->qubit_count) {
		std::cerr << "Error: QuantumCircuit::update_quantum_state(QuantumStateBase) : invalid qubit count" << std::endl;
		return;
	}

    for(const auto& gate : this->_gate_list){
        gate->update_quantum_state(state);
    }
}

void QuantumCircuit::update_quantum_state(QuantumStateBase* state, UINT start, UINT end){
	if (state->qubit_count != this->qubit_count) {
		std::cerr << "Error: QuantumCircuit::update_quantum_state(QuantumStateBase,UINT,UINT) : invalid qubit count" << std::endl;
		return;
	}
	if (start > end) {
		std::cerr << "Error: QuantumCircuit::update_quantum_state(QuantumStateBase,UINT,UINT) : start must be smaller than or equal to end" << std::endl;
		return;
	}
	if (end > this->_gate_list.size()) {
		std::cerr << "Error: QuantumCircuit::update_quantum_state(QuantumStateBase,UINT,UINT) : end must be smaller than or equal to gate_count" << std::endl;
		return;
	}
    for(UINT cursor = start ; cursor < end ; ++cursor){
        this->_gate_list[cursor]->update_quantum_state(state);
    }
}

QuantumCircuit::QuantumCircuit(const QuantumCircuit& obj):
	qubit_count(_qubit_count), gate_list(_gate_list)
{
	_gate_list.clear();
	_qubit_count = (obj.qubit_count);
	for (UINT i = 0; i < obj.gate_list.size(); ++i) {
		_gate_list.push_back(obj.gate_list[i]->copy());
	}
};


QuantumCircuit::QuantumCircuit(UINT qubit_count_):
	qubit_count(_qubit_count), gate_list(_gate_list)
{
    this->_qubit_count = qubit_count_;
}


QuantumCircuit::QuantumCircuit(std::string qasm_path, std::string qasm_loader_script_path):
qubit_count(_qubit_count), gate_list(_gate_list) {
	// generate quantum circuit from qasm
	// now we delegate compile of qasm string to quantumopencompiler in qiskit-sdk.
	std::string exec_string = std::string("python ")+qasm_loader_script_path+" "+qasm_path;
    const unsigned int MAX_BUF = 1024;
    char line[MAX_BUF];
    char* endPoint;
    FILE* fp;

    
    fp = popen(exec_string.c_str(),"r");
    if(fp==NULL){
        fprintf(stderr,"Error : cannot launch python loader or cannot load QASM: %s\n", qasm_path.c_str());
        exit(0);
    }
    fgets(line,MAX_BUF,fp);
    this->_qubit_count = atoi(line);
    while(1){
        (void)fgets(line,MAX_BUF,fp);
        if(feof(fp)) break;

        endPoint = strchr(line,'\n');
        if(endPoint != NULL) *endPoint = '\0';
        this->add_gate(gate::create_quantum_gate_from_string(line));
    }
    pclose(fp);
}

QuantumCircuit* QuantumCircuit::copy() const{
    QuantumCircuit* new_circuit = new QuantumCircuit(this->_qubit_count);
    for(const auto& gate : this->_gate_list){
        new_circuit->add_gate(gate->copy());
    }
    return new_circuit;
}

bool check_gate_index(const QuantumCircuit* circuit, const QuantumGateBase* gate) {
	auto vec1 = gate->get_target_index_list();
	auto vec2 = gate->get_control_index_list();
	UINT val = 0;
	if (vec1.size() > 0) {
		val = std::max(val, *std::max_element(vec1.begin(), vec1.end()));
	}
	if (vec2.size() > 0) {
		val = std::max(val, *std::max_element(vec2.begin(), vec2.end()));
	}
	return val < circuit->qubit_count;
}

void QuantumCircuit::add_gate(QuantumGateBase* gate){
	if (!check_gate_index(this, gate)) {
		std::cerr << "Error: QuatnumCircuit::add_gate(QuantumGateBase*): gate must be applied to qubits of which the indices are smaller than qubit_count" << std::endl;
		return;
	}
    this->_gate_list.push_back(gate);
}

void QuantumCircuit::add_gate(QuantumGateBase* gate, UINT index){
	if (!check_gate_index(this, gate)) {
		std::cerr << "Error: QuatnumCircuit::add_gate(QuantumGateBase*, UINT): gate must be applied to qubits of which the indices are smaller than qubit_count" << std::endl;
		return;
	}
	if (index > this->_gate_list.size()) {
		std::cerr << "Error: QuantumCircuit::add_gate(QuantumGateBase*, UINT) : insert index must be smaller than or equal to gate_count" << std::endl;
		return;
	}
    this->_gate_list.insert(this->_gate_list.begin() + index, gate);
}

void QuantumCircuit::add_gate_copy(const QuantumGateBase* gate){
	this->add_gate(gate->copy());
}

void QuantumCircuit::add_gate_copy(const QuantumGateBase* gate, UINT index){
	this->add_gate(gate->copy(), index);
}

void QuantumCircuit::remove_gate(UINT index){
	if (index >= this->_gate_list.size()) {
		std::cerr << "Error: QuantumCircuit::remove_gate(UINT) : index must be smaller than gate_count" << std::endl;
		return;
	}
	delete this->_gate_list[index];
    this->_gate_list.erase(this->_gate_list.begin()+index);
}

QuantumCircuit::~QuantumCircuit(){
    for(auto& gate : this->_gate_list){
        delete gate;
    }
}

bool QuantumCircuit::is_Clifford() const{
    bool flag = true;
    for(const auto& gate: this->_gate_list){
        flag = flag & gate->is_Clifford();
    }
    return flag;
}

bool QuantumCircuit::is_Gaussian() const{
    bool flag = true;
    for(const auto& gate: this->_gate_list){
        flag = flag & gate->is_Gaussian();
    }
    return flag;
}

UINT QuantumCircuit::calculate_depth() const{
    std::vector<UINT> filled_step(this->_qubit_count,0);
    UINT total_max_step = 0;
    for(const auto& gate : this->_gate_list){
        UINT max_step_among_target_qubits = 0;
        for(auto target_qubit : gate->target_qubit_list){
            max_step_among_target_qubits = std::max(max_step_among_target_qubits, filled_step[target_qubit.index()]);
        }
        for(auto control_qubit : gate->control_qubit_list){
            max_step_among_target_qubits = std::max(max_step_among_target_qubits, filled_step[control_qubit.index()]);
        }
        for (auto target_qubit : gate->target_qubit_list) {
            filled_step[target_qubit.index()] = max_step_among_target_qubits + 1;
        }
        for (auto control_qubit : gate->control_qubit_list) {
            filled_step[control_qubit.index()] = max_step_among_target_qubits + 1;
        }
        total_max_step = std::max(total_max_step , max_step_among_target_qubits + 1);
    }
    return total_max_step;
}


std::string QuantumCircuit::to_string() const {
    std::stringstream stream;
    std::vector<UINT> gate_size_count(this->_qubit_count, 0);
    UINT max_block_size = 0;

    for (const auto gate : this->_gate_list) {
        UINT whole_qubit_index_count = (UINT)(gate->target_qubit_list.size() + gate->control_qubit_list.size());
		if (whole_qubit_index_count == 0) continue;
        gate_size_count[whole_qubit_index_count - 1]++;
        max_block_size = std::max(max_block_size, whole_qubit_index_count);
    }
    stream << "*** Quantum Circuit Info ***" << std::endl;
    stream << "# of qubit: " << this->_qubit_count << std::endl;
    stream << "# of step : " << this->calculate_depth() << std::endl;
    stream << "# of gate : " << this->_gate_list.size() << std::endl;
    for (UINT i = 0; i < max_block_size; ++i) {
        stream << "# of " << i + 1 << " qubit gate: " << gate_size_count[i] << std::endl;
    }
    stream << "Clifford  : " << (this->is_Clifford() ? "yes" : "no") << std::endl;
    stream << "Gaussian  : " << (this->is_Gaussian() ? "yes" : "no") << std::endl;
    stream << std::endl;
    return stream.str();
}

std::ostream& operator<<(std::ostream& stream, const QuantumCircuit& circuit){
    stream << circuit.to_string();
    return stream;
}
std::ostream& operator<<(std::ostream& stream, const QuantumCircuit* gate) {
    stream << *gate;
    return stream;
}


void QuantumCircuit::add_X_gate(UINT target_index) {
    this->add_gate(gate::X(target_index));
}
void QuantumCircuit::add_Y_gate(UINT target_index) {
    this->add_gate(gate::Y(target_index));
}
void QuantumCircuit::add_Z_gate(UINT target_index) {
    this->add_gate(gate::Z(target_index));
}
void QuantumCircuit::add_H_gate(UINT target_index) {
    this->add_gate(gate::H(target_index));
}
void QuantumCircuit::add_S_gate(UINT target_index) {
    this->add_gate(gate::S(target_index));
}
void QuantumCircuit::add_Sdag_gate(UINT target_index) {
    this->add_gate(gate::Sdag(target_index));
}
void QuantumCircuit::add_T_gate(UINT target_index) {
    this->add_gate(gate::T(target_index));
}
void QuantumCircuit::add_Tdag_gate(UINT target_index) {
    this->add_gate(gate::Tdag(target_index));
}
void QuantumCircuit::add_sqrtX_gate(UINT target_index) {
    this->add_gate(gate::sqrtX(target_index));
}
void QuantumCircuit::add_sqrtXdag_gate(UINT target_index) {
    this->add_gate(gate::sqrtXdag(target_index));
}
void QuantumCircuit::add_sqrtY_gate(UINT target_index) {
    this->add_gate(gate::sqrtY(target_index));
}
void QuantumCircuit::add_sqrtYdag_gate(UINT target_index) {
    this->add_gate(gate::sqrtYdag(target_index));
}
void QuantumCircuit::add_P0_gate(UINT target_index) {
    this->add_gate(gate::P0(target_index));
}
void QuantumCircuit::add_P1_gate(UINT target_index) {
    this->add_gate(gate::P1(target_index));
}
void QuantumCircuit::add_CNOT_gate(UINT control_index, UINT target_index) {
    this->add_gate(gate::CNOT(control_index,target_index));
}
void QuantumCircuit::add_CZ_gate(UINT control_index, UINT target_index) {
    this->add_gate(gate::CZ(control_index,target_index));
}
void QuantumCircuit::add_SWAP_gate(UINT target_index1, UINT target_index2) {
    this->add_gate(gate::SWAP(target_index1,target_index2));
}
void QuantumCircuit::add_RX_gate(UINT target_index, double angle) {
    this->add_gate(gate::RX(target_index,angle));
}
void QuantumCircuit::add_RY_gate(UINT target_index, double angle) {
    this->add_gate(gate::RY(target_index,angle));
}
void QuantumCircuit::add_RZ_gate(UINT target_index, double angle) {
    this->add_gate(gate::RZ(target_index,angle));
}
void QuantumCircuit::add_U1_gate(UINT target_index, double phi) {
    this->add_gate(gate::U1(target_index,phi));
}
void QuantumCircuit::add_U2_gate(UINT target_index, double phi, double psi) {
    this->add_gate(gate::U2(target_index,phi,psi));
}
void QuantumCircuit::add_U3_gate(UINT target_index, double phi, double psi, double lambda) {
    this->add_gate(gate::U3(target_index,phi,psi,lambda));
}
void QuantumCircuit::add_multi_Pauli_gate(std::vector<UINT> target_index_list, std::vector<UINT> pauli_id_list) {
    this->add_gate(gate::Pauli(target_index_list, pauli_id_list));
}
void QuantumCircuit::add_multi_Pauli_gate(const PauliOperator& pauli_operator) {
    this->add_gate(gate::Pauli(pauli_operator.get_index_list(), pauli_operator.get_pauli_id_list()));
}
void QuantumCircuit::add_multi_Pauli_rotation_gate(std::vector<UINT> target_index_list, std::vector<UINT> pauli_id_list, double angle) {
    this->add_gate(gate::PauliRotation(target_index_list, pauli_id_list,angle));
}
void QuantumCircuit::add_multi_Pauli_rotation_gate(const PauliOperator& pauli_operator) {
    const double eps = 1e-14;
    if (std::abs(pauli_operator.get_coef().imag()) > eps){
        std::cerr <<  "Error: QuantumCircuit::add_multi_Pauli_rotation_gate(const PauliOperator& pauli_operator): not impremented for non hermitian" << std::endl;
    }
    this->add_gate(gate::PauliRotation(pauli_operator.get_index_list(), pauli_operator.get_pauli_id_list(), pauli_operator.get_coef().real()));
}
void QuantumCircuit::add_diagonal_observable_rotation_gate(const Observable& observable, double angle) {
    if (!observable.is_hermitian()){
        std::cerr << "Error: QuantumCircuit::add_observable_rotation_gate(const Observable& observable, double angle, UINT num_repeats): not impremented for non hermitian" << std::endl;
        return;
    }
    std::vector<PauliOperator*> operator_list = observable.get_terms();
    for (auto pauli: operator_list){
        auto pauli_rotation = gate::PauliRotation(pauli->get_index_list(), pauli->get_pauli_id_list(), pauli->get_coef().real() * angle);
        if (!pauli_rotation->is_diagonal()){
            std::cerr << "ERROR: Observable is not diagonal" << std::endl;
            return;
        }
        this->add_gate(pauli_rotation);
    }
}
void QuantumCircuit::add_observable_rotation_gate(const Observable& observable, double angle, UINT num_repeats) {
    if (!observable.is_hermitian()){
        std::cerr << "Error: QuantumCircuit::add_observable_rotation_gate(const Observable& observable, double angle, UINT num_repeats): not impremented for non hermitian" << std::endl;
        return;
    }
    UINT qubit_count_ = observable.get_qubit_count();
    std::vector<PauliOperator*> operator_list = observable.get_terms();
    if (num_repeats == 0)
        num_repeats = (UINT) std::ceil(angle * (double)qubit_count_ * 100.);
    // std::cout << num_repeats << std::endl;
    for (UINT repeat = 0; repeat < (UINT)num_repeats; ++repeat){
        for (auto pauli: operator_list){
            this->add_gate(gate::PauliRotation(pauli->get_index_list(), pauli->get_pauli_id_list(), pauli->get_coef().real() * angle/ num_repeats));
        }
    }
}
void QuantumCircuit::add_dense_matrix_gate(UINT target_index, const ComplexMatrix& matrix) {
	if (matrix.cols() != 2 || matrix.rows() != 2) {
		std::cerr << "Error: add_dense_matrix_gate(UINT, const ComplexMatrix&) : matrix must be matrix.cols()==2 and matrix.rows()==2 for single qubit gate" << std::endl;
		return;
	}

    this->add_gate(gate::DenseMatrix(target_index, matrix));
}
void QuantumCircuit::add_dense_matrix_gate(std::vector<UINT> target_index_list, const ComplexMatrix& matrix) {
	if (matrix.cols() != (1LL<<target_index_list.size()) || matrix.rows() != (1LL << target_index_list.size())) {
		std::cerr << "Error: add_dense_matrix_gate(vector<UINT>, const ComplexMatrix&) : matrix must be matrix.cols()==(1<<target_count) and matrix.rows()==(1<<target_count)" << std::endl;
		return;
	}
	
	this->add_gate(gate::DenseMatrix(target_index_list, matrix));
}

void QuantumCircuit::add_random_unitary_gate(std::vector<UINT> target_index_list) {
	this->add_gate(gate::RandomUnitary(target_index_list));
}
