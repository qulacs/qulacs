#include "gate_general.hpp"

QuantumGate_Probabilistic::QuantumGate_Probabilistic(
    const std::vector<double>& distribution,
    const std::vector<QuantumGateBase*>& gate_list)
    : _distribution(distribution) {
    if (distribution.size() != gate_list.size()) {
        throw InvalidProbabilityDistributionException(
            "Error: "
            "QuantumGate_Probabilistic::get_marginal_probability(vector<"
            "double>, vector<QuantumGateBase*>): gate_list.size() must be "
            "equal to distribution.size() or distribution.size()+1");
    }
    double sum = 0.;
    this->_cumulative_distribution.push_back(0.);
    for (auto val : distribution) {
        sum += val;
        this->_cumulative_distribution.push_back(sum);
    }
    if (sum - 1. > 1e-6) {
        throw InvalidProbabilityDistributionException(
            "Error: "
            "QuantumGate_Probabilistic::get_marginal_probability("
            "vector<double>, vector<QuantumGateBase*>): sum of "
            "probability distribution must be equal to or less than 1.0, "
            "which is " +
            std::to_string(sum));
    }
    std::transform(gate_list.cbegin(), gate_list.cend(),
        std::back_inserter(this->_gate_list),
        [](const QuantumGateBase* gate) { return gate->copy(); });
    this->_name = "Probabilistic";

    bool fullsum = (sum > 1 - 1e-6);

    if (fullsum && _gate_list.size() > 0) {
        this->_target_qubit_list = _gate_list[0]->target_qubit_list;
        this->_control_qubit_list = _gate_list[0]->control_qubit_list;
    }

    for (UINT i = (fullsum ? 1 : 0); i < _gate_list.size(); i++) {
        std::vector<TargetQubitInfo> new_target_list;
        std::vector<ControlQubitInfo> new_control_list;
        gate::get_new_qubit_list(
            this, _gate_list[i], new_target_list, new_control_list);
        this->_target_qubit_list = move(new_target_list);
        this->_control_qubit_list = move(new_control_list);
    }
    std::sort(this->_target_qubit_list.begin(), this->_target_qubit_list.end(),
        [](const TargetQubitInfo& a, const TargetQubitInfo& b) {
            return a.index() < b.index();
        });
    std::sort(this->_control_qubit_list.begin(),
        this->_control_qubit_list.end(),
        [](const ControlQubitInfo& a, const ControlQubitInfo& b) {
            return a.index() < b.index();
        });
    is_instrument = false;
};

QuantumGate_Probabilistic::QuantumGate_Probabilistic(
    const std::vector<double>& distribution,
    const std::vector<QuantumGateBase*>& gate_list,
    UINT classical_register_address)
    : QuantumGate_Probabilistic(distribution, gate_list) {
    is_instrument = true;
    _classical_register_address = classical_register_address;
}

QuantumGate_Probabilistic::~QuantumGate_Probabilistic() {
    for (unsigned int i = 0; i < _gate_list.size(); ++i) {
        delete _gate_list[i];
    }
}

void QuantumGate_Probabilistic::update_quantum_state(QuantumStateBase* state) {
    if (state->is_state_vector()) {
        double r = random.uniform();
        auto ite = std::upper_bound(_cumulative_distribution.begin(),
            _cumulative_distribution.end(), r);
        assert(ite != _cumulative_distribution.begin());
        size_t gate_index =
            std::distance(_cumulative_distribution.begin(), ite) - 1;

        if (gate_index < _gate_list.size()) {
            _gate_list[gate_index]->update_quantum_state(state);
        }
        if (is_instrument) {
            state->set_classical_value(
                this->_classical_register_address, (UINT)gate_index);
        }
    } else {
        auto org_state = state->copy();
        auto temp_state = state->copy();

        state->multiply_coef(1.0 - _cumulative_distribution.back());
        for (UINT gate_index = 0; gate_index < _gate_list.size();
             ++gate_index) {
            if (gate_index + 1 < _gate_list.size()) {
                temp_state->load(org_state);
                _gate_list[gate_index]->update_quantum_state(temp_state);
                temp_state->multiply_coef(_distribution[gate_index]);
                state->add_state(temp_state);
            } else {
                _gate_list[gate_index]->update_quantum_state(org_state);
                org_state->multiply_coef(_distribution[gate_index]);
                state->add_state(org_state);
            }
        }
        delete org_state;
        delete temp_state;
    }
}

QuantumGate_Probabilistic* QuantumGate_Probabilistic::copy() const {
    if (is_instrument) {
        return new QuantumGate_Probabilistic(
            _distribution, _gate_list, _classical_register_address);

    } else {
        return new QuantumGate_Probabilistic(_distribution, _gate_list);
    }
};

void QuantumGate_Probabilistic::set_matrix(ComplexMatrix& matrix) const {
    std::cerr << "* Warning : Gate-matrix of probabilistic gate cannot be "
                 "obtained. Identity matrix is returned."
              << std::endl;
    matrix = Eigen::MatrixXcd::Ones(1, 1);
}

boost::property_tree::ptree QuantumGate_Probabilistic::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "ProbabilisticGate");
    boost::property_tree::ptree distribution_pt;
    for (double p : _distribution) {
        boost::property_tree::ptree child;
        child.put("", p);
        distribution_pt.push_back(std::make_pair("", child));
    }
    pt.put_child("distribution", distribution_pt);
    boost::property_tree::ptree gate_list_pt;
    for (const QuantumGateBase* gate : _gate_list) {
        gate_list_pt.push_back(std::make_pair("", gate->to_ptree()));
    }
    pt.put_child("gate_list", gate_list_pt);
    if (is_instrument) {
        pt.put("is_instrument", true);
        pt.put("classical_register_address", _classical_register_address);
    } else {
        pt.put("is_instrument", false);
    }
    return pt;
}

void QuantumGate_Probabilistic::set_seed(int seed) { random.set_seed(seed); };

std::vector<double> QuantumGate_Probabilistic::get_cumulative_distribution() {
    return _cumulative_distribution;
};

std::vector<double> QuantumGate_Probabilistic::get_distribution() {
    return _distribution;
};

std::vector<QuantumGateBase*> QuantumGate_Probabilistic::get_gate_list() {
    return _gate_list;
}

void QuantumGate_Probabilistic::optimize_ProbablisticGate() {
    int n = (int)_gate_list.size();
    std::vector<std::pair<double, int>> itr;
    for (int i = 0; i < n; ++i) {
        itr.push_back(std::make_pair(_distribution[i], i));
    }
    std::sort(itr.rbegin(), itr.rend());
    std::vector<QuantumGateBase*> next_gate_list;
    for (int i = 0; i < n; ++i) {
        _distribution[i] = itr[i].first;
        next_gate_list.push_back(_gate_list[itr[i].second]);
    }
    _gate_list = next_gate_list;

    _cumulative_distribution.clear();
    double sum = 0.;
    _cumulative_distribution.push_back(0.);
    for (auto val : _distribution) {
        sum += val;
        _cumulative_distribution.push_back(sum);
    }
    return;
}

bool QuantumGate_Probabilistic::is_noise() { return true; }

QuantumGate_CPTP::QuantumGate_CPTP(std::vector<QuantumGateBase*> gate_list) {
    std::transform(gate_list.cbegin(), gate_list.cend(),
        std::back_inserter(_gate_list),
        [](const QuantumGateBase* gate) { return gate->copy(); });
    this->_name = "CPTP";

    for (UINT i = 0; i < _gate_list.size(); i++) {
        std::vector<TargetQubitInfo> new_target_list;
        std::vector<ControlQubitInfo> new_control_list;
        gate::get_new_qubit_list(
            this, _gate_list[i], new_target_list, new_control_list);
        this->_target_qubit_list = move(new_target_list);
        this->_control_qubit_list = move(new_control_list);
    }
    std::sort(this->_target_qubit_list.begin(), this->_target_qubit_list.end(),
        [](const TargetQubitInfo& a, const TargetQubitInfo& b) {
            return a.index() < b.index();
        });
    std::sort(this->_control_qubit_list.begin(),
        this->_control_qubit_list.end(),
        [](const ControlQubitInfo& a, const ControlQubitInfo& b) {
            return a.index() < b.index();
        });
    is_instrument = false;
}

QuantumGate_CPTP::QuantumGate_CPTP(
    std::vector<QuantumGateBase*> gate_list, UINT classical_register_address)
    : QuantumGate_CPTP(gate_list) {
    is_instrument = true;
    _classical_register_address = classical_register_address;
}

QuantumGate_CPTP::~QuantumGate_CPTP() {
    for (unsigned int i = 0; i < _gate_list.size(); ++i) {
        delete _gate_list[i];
    }
}

void QuantumGate_CPTP::update_quantum_state(QuantumStateBase* state) {
    if (state->is_state_vector()) {
        double r = random.uniform();

        double sum = 0.;
        double org_norm = state->get_squared_norm();

        auto buffer = state->copy();
        UINT index = 0;
        for (auto gate : _gate_list) {
            gate->update_quantum_state(buffer);
            auto norm = buffer->get_squared_norm() / org_norm;
            sum += norm;
            if (r < sum) {
                state->load(buffer);
                state->normalize(norm);
                break;
            } else {
                buffer->load(state);
                index++;
            }
        }
        if (!(r < sum)) {
            std::cerr << "* Warning : CPTP-map was not trace preserving. "
                         "Identity-map is applied."
                      << std::endl;
        }
        delete buffer;
        if (is_instrument) {
            state->set_classical_value(
                this->_classical_register_address, index);
        }
    } else {
        auto org_state = state->copy();
        auto temp_state = state->copy();
        for (UINT gate_index = 0; gate_index < _gate_list.size();
             ++gate_index) {
            if (gate_index == 0) {
                _gate_list[gate_index]->update_quantum_state(state);
            } else if (gate_index + 1 < _gate_list.size()) {
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
};

QuantumGate_CPTP* QuantumGate_CPTP::copy() const {
    if (is_instrument) {
        return new QuantumGate_CPTP(_gate_list, _classical_register_address);
    } else {
        return new QuantumGate_CPTP(_gate_list);
    }
};

void QuantumGate_CPTP::set_matrix(ComplexMatrix& matrix) const {
    std::cerr << "* Warning : Gate-matrix of CPTP-map cannot be obtained. "
                 "Identity matrix is returned."
              << std::endl;
    matrix = Eigen::MatrixXcd::Ones(1, 1);
}

boost::property_tree::ptree QuantumGate_CPTP::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "CPTPMapGate");
    boost::property_tree::ptree gate_list_pt;
    for (const QuantumGateBase* gate : _gate_list) {
        gate_list_pt.push_back(std::make_pair("", gate->to_ptree()));
    }
    pt.put_child("gate_list", gate_list_pt);
    if (is_instrument) {
        pt.put("is_instrument", true);
        pt.put("classical_register_address", _classical_register_address);
    } else {
        pt.put("is_instrument", false);
    }
    return pt;
}

std::vector<QuantumGateBase*> QuantumGate_CPTP::get_gate_list() {
    return _gate_list;
}

QuantumGate_CP::QuantumGate_CP(std::vector<QuantumGateBase*> gate_list,
    bool state_normalize, bool probability_normalize,
    bool assign_zero_if_not_matched)
    : _state_normalize(state_normalize),
      _probability_normalize(probability_normalize),
      _assign_zero_if_not_matched(assign_zero_if_not_matched) {
    std::transform(gate_list.cbegin(), gate_list.cend(),
        std::back_inserter(_gate_list),
        [](const QuantumGateBase* gate) { return gate->copy(); });
    this->_name = "CP";

    for (UINT i = 0; i < _gate_list.size(); i++) {
        std::vector<TargetQubitInfo> new_target_list;
        std::vector<ControlQubitInfo> new_control_list;
        gate::get_new_qubit_list(
            this, _gate_list[i], new_target_list, new_control_list);
        this->_target_qubit_list = move(new_target_list);
        this->_control_qubit_list = move(new_control_list);
    }
    std::sort(this->_target_qubit_list.begin(), this->_target_qubit_list.end(),
        [](const TargetQubitInfo& a, const TargetQubitInfo& b) {
            return a.index() < b.index();
        });
    std::sort(this->_control_qubit_list.begin(),
        this->_control_qubit_list.end(),
        [](const ControlQubitInfo& a, const ControlQubitInfo& b) {
            return a.index() < b.index();
        });
};

QuantumGate_CP::~QuantumGate_CP() {
    for (unsigned int i = 0; i < _gate_list.size(); ++i) {
        delete _gate_list[i];
    }
}

void QuantumGate_CP::update_quantum_state(QuantumStateBase* state) {
    if (state->is_state_vector()) {
        double r = random.uniform();

        double sum = 0.;
        double org_norm = state->get_squared_norm();

        auto buffer = state->copy();
        double norm;

        // if probability normalize = true
        //  compute sum of distribution and normalize it
        double probability_sum = 1.;
        if (_probability_normalize) {
            probability_sum = 0.;
            for (auto gate : _gate_list) {
                gate->update_quantum_state(buffer);
                norm = buffer->get_squared_norm() / org_norm;
                buffer->load(state);
                probability_sum += norm;
            }
        }

        for (auto gate : _gate_list) {
            gate->update_quantum_state(buffer);
            norm = buffer->get_squared_norm() / org_norm;
            sum += norm;
            if (r * probability_sum < sum) {
                state->load(buffer);
                if (_state_normalize) {
                    state->normalize(norm);
                }
                break;
            } else {
                buffer->load(state);
            }
        }
        if (!(r * probability_sum < sum)) {
            if (_assign_zero_if_not_matched) {
                state->multiply_coef(CPPCTYPE(0.));
            }
        }
        delete buffer;
    } else {
        auto org_state = state->copy();
        auto temp_state = state->copy();
        for (UINT gate_index = 0; gate_index < _gate_list.size();
             ++gate_index) {
            if (gate_index == 0) {
                _gate_list[gate_index]->update_quantum_state(state);
            } else if (gate_index + 1 < _gate_list.size()) {
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
};

QuantumGate_CP* QuantumGate_CP::copy() const {
    return new QuantumGate_CP(_gate_list, _state_normalize,
        _probability_normalize, _assign_zero_if_not_matched);
};

void QuantumGate_CP::set_matrix(ComplexMatrix& matrix) const {
    std::cerr << "* Warning : Gate-matrix of CP-map cannot be obtained. "
                 "Identity matrix is returned."
              << std::endl;
    matrix = Eigen::MatrixXcd::Ones(1, 1);
}

boost::property_tree::ptree QuantumGate_CP::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "CPMapGate");
    boost::property_tree::ptree gate_list_pt;
    for (const QuantumGateBase* gate : _gate_list) {
        gate_list_pt.push_back(std::make_pair("", gate->to_ptree()));
    }
    pt.put_child("gate_list", gate_list_pt);
    pt.put("state_normalize", _state_normalize);
    pt.put("probability_normalize", _probability_normalize);
    pt.put("assign_zero_if_not_matched", _assign_zero_if_not_matched);
    return pt;
}

std::vector<QuantumGateBase*> QuantumGate_CP::get_gate_list() {
    return _gate_list;
}

QuantumGate_Adaptive::QuantumGate_Adaptive(QuantumGateBase* gate,
    std::function<bool(const std::vector<UINT>&)> func_without_id)
    : _gate(gate->copy()), _func_without_id(func_without_id), _id(-1) {
    this->_name = "Adaptive";

    std::vector<TargetQubitInfo> new_target_list;
    std::vector<ControlQubitInfo> new_control_list;
    gate::get_new_qubit_list(this, _gate, new_target_list, new_control_list);
    this->_target_qubit_list = move(new_target_list);
    this->_control_qubit_list = move(new_control_list);
    std::sort(this->_target_qubit_list.begin(), this->_target_qubit_list.end(),
        [](const TargetQubitInfo& a, const TargetQubitInfo& b) {
            return a.index() < b.index();
        });
    std::sort(this->_control_qubit_list.begin(),
        this->_control_qubit_list.end(),
        [](const ControlQubitInfo& a, const ControlQubitInfo& b) {
            return a.index() < b.index();
        });
};

QuantumGate_Adaptive::QuantumGate_Adaptive(QuantumGateBase* gate,
    std::function<bool(const std::vector<UINT>&, UINT)> func_with_id, UINT id)
    : _gate(gate->copy()),
      _func_with_id(func_with_id),
      _id(static_cast<int>(id)) {
    this->_name = "Adaptive";

    // Identity gate との演算になる
    std::vector<TargetQubitInfo> new_target_list;
    std::vector<ControlQubitInfo> new_control_list;
    gate::get_new_qubit_list(this, _gate, new_target_list, new_control_list);
    this->_target_qubit_list = move(new_target_list);
    this->_control_qubit_list = move(new_control_list);
    std::sort(this->_target_qubit_list.begin(), this->_target_qubit_list.end(),
        [](const TargetQubitInfo& a, const TargetQubitInfo& b) {
            return a.index() < b.index();
        });
    std::sort(this->_control_qubit_list.begin(),
        this->_control_qubit_list.end(),
        [](const ControlQubitInfo& a, const ControlQubitInfo& b) {
            return a.index() < b.index();
        });
};

QuantumGate_Adaptive::~QuantumGate_Adaptive() { delete _gate; }

void QuantumGate_Adaptive::update_quantum_state(QuantumStateBase* state) {
    bool result;
    if (_id == -1) {
        result = _func_without_id(state->get_classical_register());
    } else {
        result = _func_with_id(state->get_classical_register(), _id);
    }
    if (result) {
        _gate->update_quantum_state(state);
    }
};

QuantumGate_Adaptive* QuantumGate_Adaptive::copy() const {
    if (_id == -1) {
        return new QuantumGate_Adaptive(_gate, _func_without_id);
    } else {
        return new QuantumGate_Adaptive(_gate, _func_with_id, _id);
    }
};

void QuantumGate_Adaptive::set_matrix(ComplexMatrix& matrix) const {
    std::cerr << "* Warning : Gate-matrix of Adaptive-gate cannot be obtained. "
                 "Identity matrix is returned."
              << std::endl;
    matrix = Eigen::MatrixXcd::Ones(1, 1);
}
