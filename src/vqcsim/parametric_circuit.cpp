#include "parametric_circuit.hpp"

#include <iostream>

#include "cppsim/gate_factory.hpp"
#include "cppsim/gate_matrix.hpp"
#include "cppsim/gate_merge.hpp"
#include "cppsim/state.hpp"
#include "cppsim/type.hpp"
#include "parametric_gate.hpp"
#include "parametric_gate_factory.hpp"

ParametricQuantumCircuit::ParametricQuantumCircuit(UINT qubit_count_)
    : QuantumCircuit(qubit_count_){};

ParametricQuantumCircuit* ParametricQuantumCircuit::copy() const {
    ParametricQuantumCircuit* new_circuit =
        new ParametricQuantumCircuit(this->qubit_count);
    for (UINT gate_pos = 0; gate_pos < this->gate_list.size(); gate_pos++) {
        auto pos = std::find(this->_parametric_gate_position.begin(),
            this->_parametric_gate_position.end(), gate_pos);
        bool is_parametric = (pos != this->_parametric_gate_position.end());

        if (is_parametric) {
            new_circuit->add_parametric_gate(
                (QuantumGate_SingleParameter*)this->gate_list[gate_pos]
                    ->copy());
        } else {
            new_circuit->add_gate(this->gate_list[gate_pos]->copy());
        }
    }
    return new_circuit;
}

void ParametricQuantumCircuit::add_parametric_gate(
    QuantumGate_SingleParameter* gate) {
    _parametric_gate_position.push_back((UINT)gate_list.size());
    this->add_gate(gate);
    _parametric_gate_list.push_back(gate);
};
void ParametricQuantumCircuit::add_parametric_gate(
    QuantumGate_SingleParameter* gate, UINT index) {
    _parametric_gate_position.push_back(index);
    this->add_gate(gate, index);
    _parametric_gate_list.push_back(gate);
}
void ParametricQuantumCircuit::add_parametric_gate_copy(
    QuantumGate_SingleParameter* gate) {
    _parametric_gate_position.push_back((UINT)gate_list.size());
    QuantumGate_SingleParameter* copied_gate = gate->copy();
    QuantumCircuit::add_gate(copied_gate);
    _parametric_gate_list.push_back(copied_gate);
};
void ParametricQuantumCircuit::add_parametric_gate_copy(
    QuantumGate_SingleParameter* gate, UINT index) {
    for (auto& val : _parametric_gate_position)
        if (val >= index) val++;
    _parametric_gate_position.push_back(index);
    QuantumGate_SingleParameter* copied_gate = gate->copy();
    QuantumCircuit::add_gate(copied_gate, index);
    _parametric_gate_list.push_back(copied_gate);
}
UINT ParametricQuantumCircuit::get_parameter_count() const {
    return (UINT)_parametric_gate_list.size();
}
double ParametricQuantumCircuit::get_parameter(UINT index) const {
    if (index >= this->_parametric_gate_list.size()) {
        std::cerr << "Error: ParametricQuantumCircuit::get_parameter(UINT): "
                     "parameter index is out of range"
                  << std::endl;
        return 0.;
    }

    return _parametric_gate_list[index]->get_parameter_value();
}
void ParametricQuantumCircuit::set_parameter(UINT index, double value) {
    if (index >= this->_parametric_gate_list.size()) {
        std::cerr
            << "Error: ParametricQuantumCircuit::set_parameter(UINT,double): "
               "parameter index is out of range"
            << std::endl;
        return;
    }

    _parametric_gate_list[index]->set_parameter_value(value);
}

std::string ParametricQuantumCircuit::to_string() const {
    std::stringstream os;
    os << QuantumCircuit::to_string();
    os << "*** Parameter Info ***" << std::endl;
    os << "# of parameter: " << this->get_parameter_count() << std::endl;
    return os.str();
}

std::ostream& operator<<(
    std::ostream& stream, const ParametricQuantumCircuit& circuit) {
    stream << circuit.to_string();
    return stream;
}
std::ostream& operator<<(
    std::ostream& stream, const ParametricQuantumCircuit* gate) {
    stream << *gate;
    return stream;
}

UINT ParametricQuantumCircuit::get_parametric_gate_position(UINT index) const {
    if (index >= this->_parametric_gate_list.size()) {
        std::cerr
            << "Error: "
               "ParametricQuantumCircuit::get_parametric_gate_position(UINT):"
               " parameter index is out of range"
            << std::endl;
        return 0;
    }

    return _parametric_gate_position[index];
}
void ParametricQuantumCircuit::add_gate(QuantumGateBase* gate) {
    QuantumCircuit::add_gate(gate);
}
void ParametricQuantumCircuit::add_gate(QuantumGateBase* gate, UINT index) {
    QuantumCircuit::add_gate(gate, index);
    for (auto& val : _parametric_gate_position)
        if (val >= index) val++;
}
void ParametricQuantumCircuit::add_gate_copy(const QuantumGateBase* gate) {
    QuantumCircuit::add_gate(gate->copy());
}
void ParametricQuantumCircuit::add_gate_copy(
    const QuantumGateBase* gate, UINT index) {
    QuantumCircuit::add_gate(gate->copy(), index);
    for (auto& val : _parametric_gate_position)
        if (val >= index) val++;
}

void ParametricQuantumCircuit::remove_gate(UINT index) {
    auto ite = std::find(_parametric_gate_position.begin(),
        _parametric_gate_position.end(), (unsigned int)index);
    if (ite != _parametric_gate_position.end()) {
        UINT dist = (UINT)std::distance(_parametric_gate_position.begin(), ite);
        _parametric_gate_position.erase(
            _parametric_gate_position.begin() + dist);
        _parametric_gate_list.erase(_parametric_gate_list.begin() + dist);
    }
    QuantumCircuit::remove_gate(index);
    for (auto& val : _parametric_gate_position)
        if (val >= index) val--;
}

void ParametricQuantumCircuit::add_parametric_RX_gate(
    UINT target_index, double initial_angle) {
    this->add_parametric_gate(gate::ParametricRX(target_index, initial_angle));
}
void ParametricQuantumCircuit::add_parametric_RY_gate(
    UINT target_index, double initial_angle) {
    this->add_parametric_gate(gate::ParametricRY(target_index, initial_angle));
}
void ParametricQuantumCircuit::add_parametric_RZ_gate(
    UINT target_index, double initial_angle) {
    this->add_parametric_gate(gate::ParametricRZ(target_index, initial_angle));
}

void ParametricQuantumCircuit::add_parametric_multi_Pauli_rotation_gate(
    std::vector<UINT> target, std::vector<UINT> pauli_id,
    double initial_angle) {
    this->add_parametric_gate(
        gate::ParametricPauliRotation(target, pauli_id, initial_angle));
}

// watle made

std::vector<double> ParametricQuantumCircuit::backprop(
    GeneralQuantumOperator* obs) {
    int n = this->qubit_count;
    QuantumState* state =
        new QuantumState(n);  //これは、ゲートを前から適用したときの状態を示す
    state->set_zero_state();
    this->update_quantum_state(state);  //一度最後までする

    QuantumState* bistate = new QuantumState(
        n);  //ニューラルネットワークのbackpropにおける、後ろからの微分値的な役目を果たす
    QuantumState* Astate = new QuantumState(n);  //一時的なやつ　

    obs->apply_to_state(Astate, *state, bistate);
    bistate->multiply_coef(-1);  //ここで、オブザーバブルの微分値が完成する。

    int m = this->gate_list.size();
    std::vector<int> gyapgp(m, -1);  // prametric gate position no gyaku
    for (UINT i = 0; i < this->get_parameter_count(); i++) {
        gyapgp[this->_parametric_gate_position[i]] = i;
    }
    std::vector<double> ans(this->get_parameter_count());

    /*
    現在、2番のゲートを見ているとする
    ゲート 0 1 2 3 4 5
         state | bistate
    前から2番までのゲートを適用した状態がAstate
    最後の微分値から逆算して3番まで　gateの逆行列を掛けたのがbistate

    1番まで掛けて、YのΘ微分した行列を掛けたやつと、　bistateの内積の実数部分をとれば答えが出ることが知られている(知られてないかも)

    ParametricR? の微分値を計算した値は、Θに180°を足した値と等しい

    だから、2番まで掛けて、 R?(π) を掛けたやつと、bistateの内積を取る

    さらに、見るゲートは逆順である。
    だから、最初にstateを最後までやって、　ゲートを進めるたびにstateとbistateに逆行列を掛けている
    */

    for (int i = m - 1; i >= 0; i--) {
        QuantumGateBase* gate_now = this->gate_list[i];  // sono gate
        if (gyapgp[i] != -1) {
            Astate->load(bistate);

            if (gate_now->get_name() == "ParametricRX") {
                QuantumGateBase* RXPI =
                    gate::RX(gate_now->get_target_index_list()[0], M_PI);
                RXPI->update_quantum_state(Astate);
                ans[gyapgp[i]] = state::inner_product(state, Astate).real();
                delete RXPI;
            } else if (gate_now->get_name() == "ParametricRY") {
                QuantumGateBase* RYPI =
                    gate::RY(gate_now->get_target_index_list()[0], M_PI);
                RYPI->update_quantum_state(Astate);
                ans[gyapgp[i]] = state::inner_product(state, Astate).real();
                delete RYPI;
            } else if (gate_now->get_name() == "ParametricRZ") {
                QuantumGateBase* RZPI =
                    gate::RZ(gate_now->get_target_index_list()[0], M_PI);
                RZPI->update_quantum_state(Astate);
                ans[gyapgp[i]] = state::inner_product(state, Astate).real();
                delete RZPI;
            } else {
                std::stringstream error_message_stream;
                error_message_stream
                    << "Error: " << gate_now->get_name()
                    << " does not support backprop in parametric";
                throw std::invalid_argument(error_message_stream.str());
            }
        }
        auto Agate = gate::get_adjoint_gate(gate_now);
        Agate->update_quantum_state(bistate);
        Agate->update_quantum_state(state);
        delete Agate;
    }
    delete Astate;
    delete state;
    delete bistate;

    return ans;
}  // CPP
