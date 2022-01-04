#pragma once

#include "gate.hpp"

class QuantumGateWrapped : public QuantumGateBase {
private:
    std::vector<QuantumGateBase*> _gate_list;
    std::vector<double> _prob_list;
    std::vector<double> _prob_cum_list;
    std::vector<UINT> _qubit_index_list;
    Random random_state;
    bool _flag_is_unital = false;
    bool _flag_save_log = false;
    std::string _reg_name = "";

    QuantumGateWrapped(MapType map_type) : QuantumGateBase(map_type){};

    void add_probabilistic_map(
        QuantumGateBase* gate, double prob, double eps = 1e-14) {
        if (prob < 0)
            throw std::invalid_argument("negative probability is assigned");
        _gate_list.push_back(gate);
        _prob_list.push_back(prob);
        double next_sum = _prob_cum_list.back() + prob;
        if (next_sum > 1.0 + eps)
            throw std::invalid_argument("sum of probability exceeds 1.0");
        _prob_cum_list.push_back(next_sum);
    }
    void update_qubit_index_list() {
        std::set<UINT> qubit_set;
        for (auto gate : _gate_list) {
            for (UINT index : gate->get_qubit_index_list()) {
                qubit_set.insert(index);
            }
        }
        _qubit_index_list.clear();
        for (auto ite = qubit_set.begin(); ite != qubit_set.end(); ++ite) {
            _qubit_index_list.push_back(*ite);
        }
    }

    void update_quantum_state_probabilistic_state_vector(
        QuantumStateBase* state) {
        double r = random_state.uniform();
        auto ite =
            std::lower_bound(_prob_cum_list.begin(), _prob_cum_list.end(), r);
        assert(ite != _prob_cum_list.begin());
        size_t gate_index = std::distance(_prob_cum_list.begin(), ite) - 1;

        if (gate_index < _gate_list.size()) {
            _gate_list[gate_index]->update_quantum_state(state);
        }
        if (_reg_name != "") state->set_classical_value(_reg_name, (int)gate_index);
    }
    void update_quantum_state_probabilistic_density_matrix(
        QuantumStateBase* state) {
        auto org_state = state->copy();
        auto temp_state = state->copy();
        for (UINT gate_index = 0; gate_index < _gate_list.size();
             ++gate_index) {
            if (gate_index == 0) {
                _gate_list[gate_index]->update_quantum_state(state);
                state->multiply_coef(_prob_list[gate_index]);
            } else if ((size_t)gate_index + 1 < _gate_list.size()) {
                temp_state->load(org_state);
                _gate_list[gate_index]->update_quantum_state(temp_state);
                temp_state->multiply_coef(_prob_list[gate_index]);
                state->add_state(temp_state);
            } else {
                _gate_list[gate_index]->update_quantum_state(org_state);
                org_state->multiply_coef(_prob_list[gate_index]);
                state->add_state(org_state);
            }
        }
        delete org_state;
        delete temp_state;
        if (_reg_name != "") state->set_classical_value(_reg_name, -1);
    }
    void update_quantum_state_CPTP_random(QuantumStateBase* state) {
        if (_reg_name != "") state->set_classical_value(_reg_name, -1);

        double r = random_state.uniform();
        double probability_sum = 0.;

        auto org_state = state->copy();
        for (UINT gate_index = 0; gate_index < _gate_list.size();
             ++gate_index) {
            _gate_list[gate_index]->update_quantum_state(state);
            double norm = state->get_squared_norm();
            probability_sum += norm;
            if (r <= probability_sum) {
                state->normalize(norm);
                if (_reg_name != "")
                    state->set_classical_value(_reg_name, gate_index);
                break;
            } else {
                state->load(org_state);
            }
        }
        delete org_state;
    }
    void update_quantum_state_CPTP_sum(QuantumStateBase* state) {
        if (_reg_name != "") state->set_classical_value(_reg_name, -1);
        auto org_state = state->copy();
        auto temp_state = state->copy();
        for (UINT gate_index = 0; gate_index < _gate_list.size();
             ++gate_index) {
            if (gate_index == 0) {
                _gate_list[gate_index]->update_quantum_state(state);
            } else if ((size_t)gate_index + 1 < _gate_list.size()) {
                temp_state->load(org_state);
                _gate_list[gate_index]->update_quantum_state(temp_state);
                state->add_state(temp_state);
            } else {
                _gate_list[gate_index]->update_quantum_state(org_state);
                state->add_state(org_state);
            }
        }
        delete org_state;
        delete temp_state;
    }

public:
    template <class Archive>
    void save(Archive& ar) const {
        ar(cereal::base_class<QuantumGateBase>(this));
        size_t size_gate_list = _gate_list.size();
        ar(CEREAL_NVP(size_gate_list));

        for (size_t i = 0; i < _gate_list.size(); ++i) {
            std::unique_ptr<QuantumGateBase> inputs;
            inputs.reset(_gate_list[i]->copy());
            ar(cereal::make_nvp("Gate " + std::to_string(i), inputs));
        }

        ar(CEREAL_NVP(_prob_list), CEREAL_NVP(_prob_cum_list),
            CEREAL_NVP(_qubit_index_list), CEREAL_NVP(_flag_is_unital),
            CEREAL_NVP(_flag_save_log), CEREAL_NVP(_reg_name));
    }

    template <class Archive>
    void load(Archive& ar) {
        ar(cereal::base_class<QuantumGateBase>(this));
        size_t size_gate_list;
        ar(CEREAL_NVP(size_gate_list));
        _gate_list.clear();

        for (size_t i = 0; i < size_gate_list; ++i) {
            std::unique_ptr<QuantumGateBase> outputs;
            ar(cereal::make_nvp("Gate " + std::to_string(i), outputs));
            _gate_list.push_back(outputs->copy());
        }

        ar(CEREAL_NVP(_prob_list), CEREAL_NVP(_prob_cum_list),
            CEREAL_NVP(_qubit_index_list), CEREAL_NVP(_flag_is_unital),
            CEREAL_NVP(_flag_save_log), CEREAL_NVP(_reg_name));
    }
    virtual std::string dump_as_byte() const override {
        // serialize quantum gate
        std::ostringstream ss;
        {
            cereal::PortableBinaryOutputArchive archive(ss);
            archive(*this);
        }
        return ss.str();
    }
    virtual void load_from_byte(std::string obj) override {
        // deserialize quantum gate
        std::istringstream ss(obj);
        {
            cereal::PortableBinaryInputArchive archive(ss);
            archive(*this);
        }
    }

    QuantumGateWrapped() = default;
    virtual ~QuantumGateWrapped() {
        for (auto& gate : _gate_list) {
            delete gate;
        }
    }
    virtual UINT get_qubit_count() const override {
        return (UINT)_qubit_index_list.size();
    }
    virtual const std::vector<UINT> get_qubit_index_list() const override {
        return _qubit_index_list;
    }
    virtual const std::vector<UINT> get_target_index_list() const override {
        return get_qubit_index_list();
    }
    virtual const std::vector<UINT> get_control_index_list() const override {
        return {};
    }
    virtual const std::vector<QuantumGateBase*>& get_kraus_list()
        const override {
        return _gate_list;
    }
    virtual void get_matrix(ComplexMatrix&) const {
        throw std::invalid_argument(
            "Get matrix is not supported in wrapper gate");
    }
    virtual std::vector<double> get_cumulative_distribution() const override {
        return _prob_cum_list;
    }

    static QuantumGateWrapped* ProbabilisticGate(
        std::vector<QuantumGateBase*> gates, const std::vector<double>& prob,
        std::string reg_name = "", bool take_ownership = false) {
        auto ptr = new QuantumGateWrapped(Probabilistic);
        ptr->_prob_list.clear();
        ptr->_prob_cum_list.clear();
        ptr->_prob_cum_list.push_back(0.);
        ptr->_reg_name = reg_name;
        for (UINT index = 0; index < gates.size(); ++index) {
            if (take_ownership)
                ptr->add_probabilistic_map(gates[index], prob[index]);
            else
                ptr->add_probabilistic_map(gates[index]->copy(), prob[index]);
        }
        ptr->update_qubit_index_list();
        return ptr;
    }
    static QuantumGateWrapped* CPTP(std::vector<QuantumGateBase*> gates,
        std::string reg_name = "", bool take_ownership = false) {
        auto ptr = new QuantumGateWrapped(MapType::CPTP);
        ptr->_prob_list.clear();
        ptr->_prob_cum_list.clear();
        ptr->_prob_cum_list.push_back(0.);
        ptr->_reg_name = reg_name;
        for (UINT index = 0; index < gates.size(); ++index) {
            if (take_ownership)
                ptr->_gate_list.push_back(gates[index]);
            else
                ptr->_gate_list.push_back(gates[index]->copy());
        }
        ptr->update_qubit_index_list();
        return ptr;
    }
    static QuantumGateWrapped* Instrument(std::vector<QuantumGateBase*> gates,
        std::string reg_name, bool take_ownership = false) {
        auto ptr = new QuantumGateWrapped(MapType::Instrument);
        ptr->_prob_list.clear();
        ptr->_prob_cum_list.clear();
        ptr->_prob_cum_list.push_back(0.);
        ptr->_reg_name = reg_name;
        for (UINT index = 0; index < gates.size(); ++index) {
            if (take_ownership)
                ptr->_gate_list.push_back(gates[index]);
            else
                ptr->_gate_list.push_back(gates[index]->copy());
        }
        ptr->update_qubit_index_list();
        return ptr;
    }
    virtual void reset_qubit_index_list(
        const std::vector<UINT>& src, const std::vector<UINT>& dst) {
        for (auto gate : _gate_list) {
            gate->reset_qubit_index_list(src, dst);
        }
        this->update_qubit_index_list();
    }

    virtual QuantumGateBase* copy() const override {
        auto ptr = new QuantumGateWrapped(this->_map_type);
        for (auto gate : _gate_list) {
            ptr->_gate_list.push_back(gate->copy());
        }
        ptr->_prob_list = _prob_list;
        ptr->_prob_cum_list = _prob_cum_list;
        ptr->_qubit_index_list = _qubit_index_list;
        ptr->_flag_is_unital = _flag_is_unital;
        ptr->_flag_save_log = _flag_save_log;
        ptr->_reg_name = _reg_name;
        return ptr;
    }
    virtual std::string to_string() const override {
        std::string s = "";
        s += "WrappedGate\n";
        s += "MapType: ";
        if (_map_type == MapType::Probabilistic)
            s += "Probabilistic";
        else if (_map_type == MapType::CPTP)
            s += "CPTP";
        else if (_map_type == MapType::Instrument)
            s += "Instrument";
        else
            throw std::invalid_argument("unknown map type");
        s += "\n";

        s += "MapCount: " + std::to_string(_gate_list.size()) + "\n";
        return s;
    }
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (_map_type == MapType::Probabilistic) {
            if (state->is_state_vector()) {
                update_quantum_state_probabilistic_state_vector(state);
            } else {
                update_quantum_state_probabilistic_density_matrix(state);
            }
        } else if (_map_type == MapType::CPTP) {
            if (state->is_state_vector()) {
                update_quantum_state_CPTP_random(state);
            } else {
                update_quantum_state_CPTP_sum(state);
            }
        } else if (_map_type == MapType::Instrument) {
            update_quantum_state_CPTP_random(state);
        } else {
            throw std::invalid_argument("Not implemented");
        }
    }
};

namespace gate {
DllExport QuantumGateWrapped* DepolarizingNoise(UINT index, double prob);
DllExport QuantumGateWrapped* TwoQubitDepolarizingNoise(
    UINT index1, UINT index2, double prob);
DllExport QuantumGateWrapped* BitFlipNoise(UINT index, double prob);
DllExport QuantumGateWrapped* DephasingNoise(UINT index, double prob);
DllExport QuantumGateWrapped* IndependentXZNoise(UINT index, double prob);
DllExport QuantumGateWrapped* AmplitudeDampingNoise(UINT index, double prob);
DllExport QuantumGateWrapped* Measurement(UINT index, std::string name);
};  // namespace gate

// Cereal Type Registration
CEREAL_REGISTER_TYPE(QuantumGateWrapped);
