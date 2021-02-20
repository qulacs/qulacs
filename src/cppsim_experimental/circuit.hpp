
/**
 * @file circuit.hpp
 *
 * @brief QuantumCircuitクラスの詳細
 */

#pragma once

#include <string>
#include <vector>

#include <cereal/access.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/complex.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

#include "gate.hpp"
#include "state.hpp"
#include "type.hpp"

/**
 * \~japanese-en 量子回路のクラス
 *
 * 量子回路を管理するクラス。QuantumGateBaseクラスをリストとして持ち、種々の操作を行う。
 * 管理する量子ゲートは量子回路の解放時にすべて解放される。
 */
class DllExport QuantumCircuit {
protected:
    std::vector<QuantumGateBase*> _gate_list;
    std::vector<QuantumGateBase*> _parametric_gate_list;
    std::vector<UINT> _parametric_gate_position;
    UINT _qubit_count;

    // prohibit shallow copy
    QuantumCircuit(const QuantumCircuit& obj);
    QuantumCircuit& operator=(const QuantumCircuit&) = delete;

    virtual void _add_gate_core(QuantumGateBase* gate, UINT index) {
        _gate_list.insert(_gate_list.begin() + index, gate);

        for (auto& val : _parametric_gate_position)
            if (val >= index) val++;
    }

public:
    template <class Archive>
    void save(Archive& ar) const {
        int size_gate_list = _gate_list.size();
        ar(CEREAL_NVP(size_gate_list));
        for (int i = 0; i < size_gate_list; ++i) {
            std::unique_ptr<QuantumGateBase> pointer;
            pointer.reset(_gate_list[i]->copy());
            ar(cereal::make_nvp("Gate "+std::to_string(i),pointer));
        }

        int size_parametric_gate_list = _parametric_gate_list.size();
        ar(CEREAL_NVP(size_parametric_gate_list));
        for (int i = 0; i < size_parametric_gate_list; ++i) {
            std::unique_ptr<QuantumGateBase> pointer;
            pointer.reset(_parametric_gate_list[i]->copy());
            ar(cereal::make_nvp("Parametric Gate "+std::to_string(i),pointer));
        }
        ar(CEREAL_NVP(_parametric_gate_position), CEREAL_NVP(_qubit_count));
    }

    template <class Archive>
    void load(Archive& ar) {
        int size_gate_list;
        ar(CEREAL_NVP(size_gate_list));
        _gate_list.clear();
        for (int i = 0; i < size_gate_list; ++i) {
            std::unique_ptr<QuantumGateBase> pointer;
            ar(cereal::make_nvp("Gate "+std::to_string(i),pointer));
            _gate_list.push_back(pointer->copy());
        }

        int size_parametric_gate_list;
        ar(CEREAL_NVP(size_parametric_gate_list));
        _parametric_gate_list.clear();
        for (int i = 0; i < size_parametric_gate_list; ++i) {
            std::unique_ptr<QuantumGateBase> pointer;
            ar(cereal::make_nvp("Parametric Gate " + std::to_string(i),pointer));
            _parametric_gate_list.push_back(pointer->copy());
        }

        ar(CEREAL_NVP(_parametric_gate_position), CEREAL_NVP(_qubit_count));
    }

    virtual UINT get_qubit_count() const { return this->_qubit_count; }
    virtual const std::vector<QuantumGateBase*>& get_gate_list() const {
        return this->_gate_list;
    }

    QuantumCircuit(UINT qubit_count) : _qubit_count(qubit_count) {}
    virtual ~QuantumCircuit() {
        for (auto ptr : _gate_list) {
            delete ptr;
        }
    }

    QuantumCircuit* copy() const {
        QuantumCircuit* new_circuit = new QuantumCircuit(this->_qubit_count);
        for (const auto& gate : this->_gate_list) {
            new_circuit->add_gate(gate->copy());
        }
        return new_circuit;
    }

    ////////////////////// BASIC CONTROL OF QUANTUM CIRCUIT
    virtual void add_gate(const QuantumGateBase* gate) {
        _add_gate_core(gate->copy(), (UINT)_gate_list.size());
    };
    virtual void add_gate(const QuantumGateBase* gate, UINT index) {
        _add_gate_core(gate->copy(), index);
    };
    virtual void add_gate_take(QuantumGateBase* gate) {
        _add_gate_core(gate, (UINT)_gate_list.size());
    };
    virtual void add_gate_take(QuantumGateBase* gate, UINT index) {
        _add_gate_core(gate, index);
    };

    virtual void replace_gate(const QuantumGateBase* gate, UINT index) {
        remove_gate(index);
        add_gate(gate, index);
    }
    virtual void replace_gate_take(QuantumGateBase* gate, UINT index) {
        remove_gate(index);
        add_gate_take(gate, index);
    }
    virtual void merge_circuit(const QuantumCircuit* circuit) {
        if (this->get_qubit_count() != circuit->get_qubit_count()) {
            throw std::invalid_argument(
                "Error: "
                "QuantumCircuit::add_circuit(QuantumCircuit*):"
                "Qubit count doesn't match!");
        }
        for (auto gate : circuit->get_gate_list()) {
            this->add_gate(gate);
        }
        return;
    }
    virtual void remove_gate(UINT index) {
        delete _gate_list.at(index);
        _gate_list.erase(_gate_list.begin() + index);

        auto ite = std::find(_parametric_gate_position.begin(),
            _parametric_gate_position.end(), index);
        if (ite != _parametric_gate_position.end()) {
            UINT dist =
                (UINT)std::distance(_parametric_gate_position.begin(), ite);
            _parametric_gate_position.erase(
                _parametric_gate_position.begin() + dist);
            _parametric_gate_list.erase(_parametric_gate_list.begin() + dist);
        }
        for (auto& val : _parametric_gate_position)
            if (val >= index) val--;
    }

    virtual void add_noise_gate(const QuantumGateBase* org_gate,
        std::string noise_type, double noise_prob) {
        QuantumGateBase* gate = org_gate->copy();
        this->add_gate(gate);
        auto itr = gate->get_qubit_index_list();
        if (noise_type == "Depolarizing") {
            if (itr.size() == 1) {
                this->add_gate(gate::DepolarizingNoise(itr[0], noise_prob));
            } else if (itr.size() == 2) {
                this->add_gate(gate::TwoQubitDepolarizingNoise(
                    itr[0], itr[1], noise_prob));
            } else {
                throw std::invalid_argument(
                    "Error: "
                    "QuantumCircuit::add_noise_gate(QuantumGateBase*,"
                    "string,double) : "
                    "depolarizing noise can be used up to 2 qubits, but "
                    "this gate has " +
                    std::to_string(itr.size()) + " qubits.");
                /*
                std::cerr
                    << "Error: "
                       "QuantumCircuit::add_noise_gate(QuantumGateBase*,"
                       "string,double) : "
                       "depolarizing noise can be used up to 2 qubits, but "
                       "this gate has "
                    << itr.size() << " qubits." << std::endl;
               */
            }
        } else if (noise_type == "BitFlip") {
            if (itr.size() == 1) {
                this->add_gate(gate::BitFlipNoise(itr[0], noise_prob));
            } else {
                throw std::invalid_argument(
                    "Error: "
                    "QuantumCircuit::add_noise_gate(QuantumGateBase*,string,"
                    "double) : "
                    "BitFlip noise can be used by 1 qubits, but this gate "
                    "has " +
                    std::to_string(itr.size()) + " qubits.");
                /*
                std::cerr
                    << "Error: "
                       "QuantumCircuit::add_noise_gate(QuantumGateBase*,string,"
                       "double) : "
                       "BitFlip noise can be used by 1 qubits, but this gate "
                       "has "
                    << itr.size() << " qubits." << std::endl;
                */
            }
        } else if (noise_type == "Dephasing") {
            if (itr.size() == 1) {
                this->add_gate(gate::DephasingNoise(itr[0], noise_prob));
            } else {
                throw std::invalid_argument(
                    "Error: "
                    "QuantumCircuit::add_noise_gate(QuantumGateBase*,string,"
                    "double) : "
                    "Dephasing noise can be used by 1 qubits, but this gate "
                    "has " +
                    std::to_string(itr.size()) + " qubits.");
                /*
                std::cerr
                    << "Error: "
                       "QuantumCircuit::add_noise_gate(QuantumGateBase*,string,"
                       "double) : "
                       "Dephasing noise can be used by 1 qubits, but this gate "
                       "has "
                    << itr.size() << " qubits." << std::endl;
                */
            }
        } else if (noise_type == "IndependentXZ") {
            if (itr.size() == 1) {
                this->add_gate(gate::IndependentXZNoise(itr[0], noise_prob));
            } else {
                throw std::invalid_argument(
                    "Error: "
                    "QuantumCircuit::add_noise_gate(QuantumGateBase*,string,"
                    "double) : "
                    "IndependentXZ noise can be used by 1 qubits, but this "
                    "gate "
                    "has " +
                    std::to_string(itr.size()) + " qubits.");
                /*
                std::cerr << "Error: "
                             "QuantumCircuit::add_noise_gate(QuantumGateBase*,"
                             "string,double) : "
                             "IndependentXZ noise can be used by 1 qubits, but "
                             "this gate has "
                          << itr.size() << " qubits." << std::endl;
                */
            }
        }
        /*
        else if (noise_type == "AmplitudeDamping") {
            if (itr.size() == 1) {
                this->add_gate(gate::AmplitudeDampingNoise(itr[0], noise_prob));
            }
            else {
                std::cerr
                    << "Error: "
                    "QuantumCircuit::add_noise_gate(QuantumGateBase*,string,"
                    "double) : AmplitudeDamping noise can be used by 1 qubits, "
                    "but this gate has "
                    << itr.size() << " qubits." << std::endl;
            }
        }
        */
        else {
            throw std::invalid_argument(
                "Error: "
                "QuantumCircuit::add_noise_gate(QuantumGateBase*,string,"
                "double) : noise_type is undetectable. your noise_type = '" +
                noise_type);
            /*
            std::cerr
                << "Error: "
                   "QuantumCircuit::add_noise_gate(QuantumGateBase*,string,"
                   "double) : noise_type is undetectable. your noise_type = '"
                << noise_type << "'." << std::endl;
            */
        }
    }

    /// PARAMETER MANGE
    virtual void add_parametric_gate(const QuantumGateBase* gate) {
        add_parametric_gate(gate, (UINT)_gate_list.size());
    }
    virtual void add_parametric_gate(const QuantumGateBase* gate, UINT index) {
        for (auto& val : _parametric_gate_position)
            if (val >= index) val++;
        _parametric_gate_position.push_back(index);
        add_gate(gate, index);
        _parametric_gate_list.push_back(_gate_list.at(index));
    }

    virtual UINT get_parameter_count() const {
        return (UINT)_parametric_gate_list.size();
    }
    virtual double get_parameter(UINT index) const {
        return _parametric_gate_list.at(index)->get_parameter("");
    }
    virtual void set_parameter(UINT index, double value) {
        _parametric_gate_list.at(index)->set_parameter("", value);
    }
    virtual UINT get_parametric_gate_position(UINT index) const {
        return _parametric_gate_position.at(index);
    }

    /////////////////////////////// UPDATE QUANTUM STATE
    void update_quantum_state(QuantumStateBase* state) {
        update_quantum_state(state, 0, (UINT)_gate_list.size());
    };
    void update_quantum_state(
        QuantumStateBase* state, UINT start_index, UINT end_index) {
        for (UINT index = start_index; index < end_index; ++index) {
            _gate_list.at(index)->update_quantum_state(state);
        }
    }

    /////////////////////////////// CHECK PROPERTY OF QUANTUM CIRCUIT
    UINT calculate_depth() const {
        std::vector<UINT> filled_step(this->_qubit_count, 0);
        UINT total_max_step = 0;
        for (const auto& gate : this->_gate_list) {
            UINT max_step_among_target_qubits = 0;
            std::vector<UINT> whole_qubit = gate->get_qubit_index_list();

            for (auto index : whole_qubit) {
                max_step_among_target_qubits =
                    std::max(max_step_among_target_qubits, filled_step[index]);
            }
            for (auto index : whole_qubit) {
                filled_step[index] = max_step_among_target_qubits + 1;
            }
            total_max_step =
                std::max(total_max_step, max_step_among_target_qubits + 1);
        }
        return total_max_step;
    }

    virtual std::string to_string() const {
        std::stringstream stream;
        std::vector<UINT> gate_size_count(this->_qubit_count, 0);
        UINT max_block_size = 0;

        for (const auto gate : this->_gate_list) {
            UINT whole_qubit_index_count = gate->get_qubit_count();
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
        stream << std::endl;
        return stream.str();
    }
    friend DllExport std::ostream& operator<<(
        std::ostream& os, const QuantumCircuit& circuit) {
        os << circuit.to_string() << std::endl;
        return os;
    }
};
