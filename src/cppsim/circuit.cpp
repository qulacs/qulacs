#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "circuit.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "exception.hpp"
#include "gate.hpp"
#include "gate_factory.hpp"
#include "gate_matrix.hpp"
#include "observable.hpp"
#include "pauli_operator.hpp"

bool check_gate_index(
    const QuantumCircuit* circuit, const QuantumGateBase* gate);

void QuantumCircuit::update_quantum_state(QuantumStateBase* state) {
    if (state->qubit_count != this->qubit_count) {
        throw InvalidQubitCountException(
            "Error: "
            "QuantumCircuit::update_quantum_state(QuantumStateBase) : "
            "invalid qubit count");
    }

    for (const auto& gate : this->_gate_list) {
        gate->update_quantum_state(state);
    }
}

void QuantumCircuit::update_quantum_state(QuantumStateBase* state, UINT seed) {
    Random random;
    random.set_seed(seed);
    this->update_quantum_state(state);
}

void QuantumCircuit::update_quantum_state(
    QuantumStateBase* state, UINT start, UINT end) {
    if (state->qubit_count != this->qubit_count) {
        throw InvalidQubitCountException(
            "Error: "
            "QuantumCircuit::update_quantum_state(QuantumStateBase,UINT,"
            "UINT) : invalid qubit count");
    }
    if (start > end) {
        throw GateIndexOutOfRangeException(
            "Error: "
            "QuantumCircuit::update_quantum_state(QuantumStateBase,UINT,"
            "UINT) : start must be smaller than or equal to end");
    }
    if (end > this->_gate_list.size()) {
        throw GateIndexOutOfRangeException(
            "Error: "
            "QuantumCircuit::update_quantum_state(QuantumStateBase,UINT,"
            "UINT) : end must be smaller than or equal to gate_count");
    }
    for (UINT cursor = start; cursor < end; ++cursor) {
        this->_gate_list[cursor]->update_quantum_state(state);
    }
}

void QuantumCircuit::update_quantum_state(
    QuantumStateBase* state, UINT start, UINT end, UINT seed) {
    Random random;
    random.set_seed(seed);
    this->update_quantum_state(state, start, end);
}

QuantumCircuit::QuantumCircuit(const QuantumCircuit& obj)
    : qubit_count(_qubit_count), gate_list(_gate_list) {
    _gate_list.clear();
    _qubit_count = (obj.qubit_count);
    for (UINT i = 0; i < obj.gate_list.size(); ++i) {
        _gate_list.push_back(obj.gate_list[i]->copy());
    }
};

QuantumCircuit::QuantumCircuit(UINT qubit_count_)
    : qubit_count(_qubit_count), gate_list(_gate_list) {
    this->_qubit_count = qubit_count_;
}

QuantumCircuit* QuantumCircuit::copy() const {
    QuantumCircuit* new_circuit = new QuantumCircuit(this->_qubit_count);
    for (const auto& gate : this->_gate_list) {
        new_circuit->add_gate(gate->copy());
    }
    return new_circuit;
}

bool check_gate_index(
    const QuantumCircuit* circuit, const QuantumGateBase* gate) {
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

void QuantumCircuit::add_gate(QuantumGateBase* gate) {
    if (!check_gate_index(this, gate)) {
        throw InvalidQubitCountException(
            "Error: QuatnumCircuit::add_gate(QuantumGateBase*): gate "
            "must be "
            "applied to qubits of which the indices are smaller than "
            "qubit_count");
    }
    this->_gate_list.push_back(gate);
}

void QuantumCircuit::add_gate(QuantumGateBase* gate, UINT index) {
    if (!check_gate_index(this, gate)) {
        throw InvalidQubitCountException(
            "Error: QuatnumCircuit::add_gate(QuantumGateBase*, UINT): "
            "gate must be applied to qubits of which the indices are "
            "smaller than qubit_count");
    }
    if (index > this->_gate_list.size()) {
        throw GateIndexOutOfRangeException(
            "Error: QuantumCircuit::add_gate(QuantumGateBase*, UINT) : "
            "insert index must be smaller than or equal to gate_count");
    }
    this->_gate_list.insert(this->_gate_list.begin() + index, gate);
}

void QuantumCircuit::add_gate_copy(const QuantumGateBase* gate) {
    this->add_gate(gate->copy());
}

void QuantumCircuit::add_gate_copy(const QuantumGateBase* gate, UINT index) {
    this->add_gate(gate->copy(), index);
}

void QuantumCircuit::add_noise_gate(
    QuantumGateBase* gate, std::string noise_type, double noise_prob) {
    this->add_gate(gate);
    auto vec1 = gate->get_target_index_list();
    auto vec2 = gate->get_control_index_list();
    std::vector<UINT> itr = vec1;
    for (auto x : vec2) {
        itr.push_back(x);
    }
    if (noise_type == "Depolarizing") {
        if (itr.size() == 1) {
            this->add_gate(gate::DepolarizingNoise(itr[0], noise_prob));
        } else if (itr.size() == 2) {
            this->add_gate(
                gate::TwoQubitDepolarizingNoise(itr[0], itr[1], noise_prob));
        } else {
            throw InvalidQubitCountException(
                "Error: "
                "QuantumCircuit::add_noise_gate(QuantumGateBase*,"
                "string,double) : "
                "depolarizing noise can be used up to 2 qubits, but "
                "this gate has " +
                std::to_string(itr.size()) + " qubits.");
        }
    } else if (noise_type == "BitFlip") {
        if (itr.size() == 1) {
            this->add_gate(gate::BitFlipNoise(itr[0], noise_prob));
        } else {
            throw InvalidQubitCountException(
                "Error: "
                "QuantumCircuit::add_noise_gate(QuantumGateBase*,string,"
                "double) : "
                "BitFlip noise can be used by 1 qubits, but this gate has " +
                std::to_string(itr.size()) + " qubits.");
        }
    } else if (noise_type == "Dephasing") {
        if (itr.size() == 1) {
            this->add_gate(gate::DephasingNoise(itr[0], noise_prob));
        } else {
            throw InvalidQubitCountException(
                "Error: "
                "QuantumCircuit::add_noise_gate(QuantumGateBase*,string,"
                "double) : "
                "Dephasing noise can be used by 1 qubits, but this gate has " +
                std::to_string(itr.size()) + " qubits.");
        }
    } else if (noise_type == "IndependentXZ") {
        if (itr.size() == 1) {
            this->add_gate(gate::IndependentXZNoise(itr[0], noise_prob));
        } else {
            throw InvalidQubitCountException(
                "Error: "
                "QuantumCircuit::add_noise_gate(QuantumGateBase*,"
                "string,double) : "
                "IndependentXZ noise can be used by 1 qubits, but "
                "this gate has " +
                std::to_string(itr.size()) + " qubits.");
        }
    } else if (noise_type == "AmplitudeDamping") {
        if (itr.size() == 1) {
            this->add_gate(gate::AmplitudeDampingNoise(itr[0], noise_prob));
        } else {
            throw InvalidQubitCountException(
                "Error: "
                "QuantumCircuit::add_noise_gate(QuantumGateBase*,string,"
                "double) : AmplitudeDamping noise can be used by 1 qubits, "
                "but this gate has " +
                std::to_string(itr.size()) + " qubits.");
        }
    } else {
        throw InvalidNoiseTypeIdentifierException(
            "Error: "
            "QuantumCircuit::add_noise_gate(QuantumGateBase*,string,"
            "double) : noise_type is undetectable. your noise_type = '" +
            noise_type + "'.");
    }
}

void QuantumCircuit::add_noise_gate_copy(
    QuantumGateBase* gate, std::string noise_type, double noise_prob) {
    this->add_noise_gate(gate->copy(), noise_type, noise_prob);
}

void QuantumCircuit::remove_gate(UINT index) {
    if (index >= this->_gate_list.size()) {
        throw GateIndexOutOfRangeException(
            "Error: QuantumCircuit::remove_gate(UINT) : index must be "
            "smaller than gate_count");
    }
    delete this->_gate_list[index];
    this->_gate_list.erase(this->_gate_list.begin() + index);
}

void QuantumCircuit::move_gate(UINT from_index, UINT to_index) {
    if (from_index >= this->_gate_list.size() ||
        to_index >= this->_gate_list.size()) {
        throw GateIndexOutOfRangeException(
            "Error: QuantumCircuit::move_gate(UINT, UINT) : "
            "index must be smaller than gate_count");
    }
    if (from_index < to_index) {
        std::rotate(this->_gate_list.begin() + from_index,
            this->_gate_list.begin() + from_index + 1,
            this->_gate_list.begin() + to_index + 1);
    } else {
        std::rotate(this->_gate_list.rbegin() +
                        (this->_gate_list.size() - from_index - 1),
            this->_gate_list.rbegin() + (this->_gate_list.size() - from_index),
            this->_gate_list.rbegin() + (this->_gate_list.size() - to_index));
    }
}

QuantumCircuit::~QuantumCircuit() {
    for (auto& gate : this->_gate_list) {
        delete gate;
    }
}

bool QuantumCircuit::is_Clifford() const {
    return std::all_of(this->_gate_list.cbegin(), this->_gate_list.cend(),
        [](const QuantumGateBase* gate) { return gate->is_Clifford(); });
}

bool QuantumCircuit::is_Gaussian() const {
    return std::all_of(this->_gate_list.cbegin(), this->_gate_list.cend(),
        [](const QuantumGateBase* gate) { return gate->is_Gaussian(); });
}

UINT QuantumCircuit::calculate_depth() const {
    std::vector<UINT> filled_step(this->_qubit_count, 0);
    UINT total_max_step = 0;
    for (const auto& gate : this->_gate_list) {
        UINT max_step_among_target_qubits = 0;
        for (auto target_qubit : gate->target_qubit_list) {
            max_step_among_target_qubits =
                std::max(max_step_among_target_qubits,
                    filled_step[target_qubit.index()]);
        }
        for (auto control_qubit : gate->control_qubit_list) {
            max_step_among_target_qubits =
                std::max(max_step_among_target_qubits,
                    filled_step[control_qubit.index()]);
        }
        for (auto target_qubit : gate->target_qubit_list) {
            filled_step[target_qubit.index()] =
                max_step_among_target_qubits + 1;
        }
        for (auto control_qubit : gate->control_qubit_list) {
            filled_step[control_qubit.index()] =
                max_step_among_target_qubits + 1;
        }
        total_max_step =
            std::max(total_max_step, max_step_among_target_qubits + 1);
    }
    return total_max_step;
}

std::string QuantumCircuit::to_string() const {
    std::stringstream stream;
    std::vector<UINT> gate_size_count(this->_qubit_count, 0);
    UINT max_block_size = 0;

    for (const auto gate : this->_gate_list) {
        UINT whole_qubit_index_count = (UINT)(
            gate->target_qubit_list.size() + gate->control_qubit_list.size());
        if (whole_qubit_index_count == 0) continue;
        gate_size_count[whole_qubit_index_count - 1]++;
        max_block_size = std::max(max_block_size, whole_qubit_index_count);
    }
    stream << "*** Quantum Circuit Info ***" << std::endl;
    stream << "# of qubit: " << this->_qubit_count << std::endl;
    stream << "# of step : " << this->calculate_depth() << std::endl;
    stream << "# of gate : " << this->_gate_list.size() << std::endl;
    for (UINT i = 0; i < max_block_size; ++i) {
        stream << "# of " << i + 1 << " qubit gate: " << gate_size_count[i]
               << std::endl;
    }
    stream << "Clifford  : " << (this->is_Clifford() ? "yes" : "no")
           << std::endl;
    stream << "Gaussian  : " << (this->is_Gaussian() ? "yes" : "no")
           << std::endl;
    stream << std::endl;
    return stream.str();
}

std::ostream& operator<<(std::ostream& stream, const QuantumCircuit& circuit) {
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
    this->add_gate(gate::CNOT(control_index, target_index));
}
void QuantumCircuit::add_CZ_gate(UINT control_index, UINT target_index) {
    this->add_gate(gate::CZ(control_index, target_index));
}
void QuantumCircuit::add_SWAP_gate(UINT target_index1, UINT target_index2) {
    this->add_gate(gate::SWAP(target_index1, target_index2));
}
void QuantumCircuit::add_FusedSWAP_gate(
    UINT target_index1, UINT target_index2, UINT block_size) {
    this->add_gate(gate::FusedSWAP(target_index1, target_index2, block_size));
}
void QuantumCircuit::add_RX_gate(UINT target_index, double angle) {
    this->add_gate(gate::RX(target_index, angle));
}
void QuantumCircuit::add_RY_gate(UINT target_index, double angle) {
    this->add_gate(gate::RY(target_index, angle));
}
void QuantumCircuit::add_RZ_gate(UINT target_index, double angle) {
    this->add_gate(gate::RZ(target_index, angle));
}
void QuantumCircuit::add_RotInvX_gate(UINT target_index, double angle) {
    this->add_gate(gate::RotInvX(target_index, angle));
}
void QuantumCircuit::add_RotInvY_gate(UINT target_index, double angle) {
    this->add_gate(gate::RotInvY(target_index, angle));
}
void QuantumCircuit::add_RotInvZ_gate(UINT target_index, double angle) {
    this->add_gate(gate::RotInvZ(target_index, angle));
}
void QuantumCircuit::add_RotX_gate(UINT target_index, double angle) {
    this->add_gate(gate::RotX(target_index, angle));
}
void QuantumCircuit::add_RotY_gate(UINT target_index, double angle) {
    this->add_gate(gate::RotY(target_index, angle));
}
void QuantumCircuit::add_RotZ_gate(UINT target_index, double angle) {
    this->add_gate(gate::RotZ(target_index, angle));
}
void QuantumCircuit::add_U1_gate(UINT target_index, double lambda) {
    this->add_gate(gate::U1(target_index, lambda));
}
void QuantumCircuit::add_U2_gate(UINT target_index, double phi, double lambda) {
    this->add_gate(gate::U2(target_index, phi, lambda));
}
void QuantumCircuit::add_U3_gate(
    UINT target_index, double theta, double phi, double lambda) {
    this->add_gate(gate::U3(target_index, theta, phi, lambda));
}
void QuantumCircuit::add_multi_Pauli_gate(
    std::vector<UINT> target_index_list, std::vector<UINT> pauli_id_list) {
    this->add_gate(gate::Pauli(target_index_list, pauli_id_list));
}
void QuantumCircuit::add_multi_Pauli_gate(const PauliOperator& pauli_operator) {
    this->add_gate(gate::Pauli(
        pauli_operator.get_index_list(), pauli_operator.get_pauli_id_list()));
}
void QuantumCircuit::add_multi_Pauli_rotation_gate(
    std::vector<UINT> target_index_list, std::vector<UINT> pauli_id_list,
    double angle) {
    this->add_gate(
        gate::PauliRotation(target_index_list, pauli_id_list, angle));
}
void QuantumCircuit::add_multi_Pauli_rotation_gate(
    const PauliOperator& pauli_operator) {
    const double eps = 1e-14;
    if (std::abs(pauli_operator.get_coef().imag()) > eps) {
        throw NonHermitianException(
            "Error: QuantumCircuit::add_multi_Pauli_rotation_gate(const "
            "PauliOperator& pauli_operator): not implemented for non "
            "hermitian");
    }
    this->add_gate(gate::PauliRotation(pauli_operator.get_index_list(),
        pauli_operator.get_pauli_id_list(), pauli_operator.get_coef().real()));
}
void QuantumCircuit::add_diagonal_observable_rotation_gate(
    const Observable& observable, double angle) {
    if (!observable.is_hermitian()) {
        throw NonHermitianException(
            "Error: QuantumCircuit::add_observable_rotation_gate(const "
            "Observable& observable, double angle, UINT num_repeats): not "
            "implemented for non hermitian");
    }
    std::vector<PauliOperator*> operator_list = observable.get_terms();
    for (auto pauli : operator_list) {
        auto pauli_rotation = gate::PauliRotation(pauli->get_index_list(),
            pauli->get_pauli_id_list(), pauli->get_coef().real() * angle);
        if (!pauli_rotation->is_diagonal()) {
            throw InvalidObservableException(
                "ERROR: Observable is not diagonal");
        }
        this->add_gate(pauli_rotation);
    }
}
void QuantumCircuit::add_observable_rotation_gate(
    const Observable& observable, double angle, UINT num_repeats) {
    if (!observable.is_hermitian()) {
        throw NonHermitianException(
            "Error: QuantumCircuit::add_observable_rotation_gate(const "
            "Observable& observable, double angle, UINT num_repeats): not "
            "implemented for non hermitian");
    }
    UINT qubit_count_ = observable.get_qubit_count();
    std::vector<PauliOperator*> operator_list = observable.get_terms();
    if (num_repeats == 0)
        num_repeats =
            static_cast<UINT>(std::ceil(angle * (double)qubit_count_ * 100.));
    // std::cout << num_repeats << std::endl;
    for (UINT repeat = 0; repeat < num_repeats; ++repeat) {
        for (auto pauli : operator_list) {
            this->add_gate(gate::PauliRotation(pauli->get_index_list(),
                pauli->get_pauli_id_list(),
                pauli->get_coef().real() * angle / num_repeats));
        }
    }
}
void QuantumCircuit::add_dense_matrix_gate(
    UINT target_index, const ComplexMatrix& matrix) {
    if (matrix.cols() != 2 || matrix.rows() != 2) {
        throw InvalidMatrixGateSizeException(
            "Error: add_dense_matrix_gate(UINT, const ComplexMatrix&) "
            ": matrix "
            "must be matrix.cols()==2 and matrix.rows()==2 for single "
            "qubit gate");
    }

    this->add_gate(gate::DenseMatrix(target_index, matrix));
}
void QuantumCircuit::add_dense_matrix_gate(
    std::vector<UINT> target_index_list, const ComplexMatrix& matrix) {
    if (matrix.cols() != (1LL << target_index_list.size()) ||
        matrix.rows() != (1LL << target_index_list.size())) {
        throw InvalidMatrixGateSizeException(
            "Error: add_dense_matrix_gate(vector<UINT>, const "
            "ComplexMatrix&) : "
            "matrix must be matrix.cols()==(1<<target_count) and "
            "matrix.rows()==(1<<target_count)");
    }

    this->add_gate(gate::DenseMatrix(target_index_list, matrix));
}

void QuantumCircuit::add_random_unitary_gate(
    std::vector<UINT> target_index_list) {
    this->add_gate(gate::RandomUnitary(target_index_list));
}

void QuantumCircuit::add_random_unitary_gate(
    std::vector<UINT> target_index_list, UINT seed) {
    this->add_gate(gate::RandomUnitary(target_index_list, seed));
}

boost::property_tree::ptree QuantumCircuit::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "QuantumCircuit");
    pt.put("qubit_count", _qubit_count);
    boost::property_tree::ptree gate_list_pt;
    for (const QuantumGateBase* gate : _gate_list) {
        gate_list_pt.push_back(std::make_pair("", gate->to_ptree()));
    }
    pt.put_child("gate_list", gate_list_pt);
    return pt;
}

QuantumCircuit* QuantumCircuit::get_inverse(void) {
    auto ans = new QuantumCircuit(this->qubit_count);
    for (auto itr = std::rbegin(this->_gate_list);
         itr != std::rend(this->_gate_list); ++itr) {
        ans->add_gate((*itr)->get_inverse());
    }
    return ans;
}

namespace circuit {
QuantumCircuit* from_ptree(const boost::property_tree::ptree& pt) {
    std::string name = pt.get<std::string>("name");
    if (name != "QuantumCircuit") {
        throw UnknownPTreePropertyValueException(
            "unknown value for property \"name\":" + name);
    }
    UINT qubit_count = pt.get<UINT>("qubit_count");
    QuantumCircuit* circuit = new QuantumCircuit(qubit_count);
    for (const boost::property_tree::ptree::value_type& gate_pair :
        pt.get_child("gate_list")) {
        QuantumGateBase* gate = gate::from_ptree(gate_pair.second);
        circuit->add_gate(gate);
    }
    return circuit;
}
}  // namespace circuit
