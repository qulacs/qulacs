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

    QuantumGateWrapped(MapType map_type) : QuantumGateBase(map_type) {};

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
        bool take_ownership = false) {
        auto ptr = new QuantumGateWrapped(Probabilistic);
        ptr->_prob_list.clear();
        ptr->_prob_cum_list.clear();
        ptr->_prob_cum_list.push_back(0.);
        for (UINT index = 0; index < gates.size(); ++index) {
            if (take_ownership)
                ptr->add_probabilistic_map(gates[index], prob[index]);
            else
                ptr->add_probabilistic_map(gates[index]->copy(), prob[index]);
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
        return "WrappedGate (TODO)";
    }
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (_map_type == Probabilistic) {
            if (state->is_state_vector()) {
                double r = random_state.uniform();
                auto ite = std::lower_bound(
                    _prob_cum_list.begin(), _prob_cum_list.end(), r);
                assert(ite != _prob_cum_list.begin());
                size_t gate_index =
                    std::distance(_prob_cum_list.begin(), ite) - 1;

                if (gate_index < _gate_list.size()) {
                    _gate_list[gate_index]->update_quantum_state(state);
                }
            }
            else {
                auto org_state = state->copy();
                auto temp_state = state->copy();
                for (UINT gate_index = 0; gate_index < _gate_list.size();
                    ++gate_index) {
                    if (gate_index == 0) {
                        _gate_list[gate_index]->update_quantum_state(state);
                        state->multiply_coef(_prob_list[gate_index]);
                    }
                    else if (gate_index + 1 < _gate_list.size()) {
                        temp_state->load(org_state);
                        _gate_list[gate_index]->update_quantum_state(
                            temp_state);
                        temp_state->multiply_coef(_prob_list[gate_index]);
                        state->add_state(temp_state);
                    }
                    else {
                        _gate_list[gate_index]->update_quantum_state(org_state);
                        org_state->multiply_coef(_prob_list[gate_index]);
                        state->add_state(org_state);
                    }
                }
                delete org_state;
                delete temp_state;
            }
        }
        else {
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
};  // namespace gate

// Cereal Type Registration
CEREAL_REGISTER_TYPE(QuantumGateWrapped);
