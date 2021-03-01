#pragma once

#include "gate.hpp"
#include "state.hpp"
#include "utility.hpp"

/**
 * \~japanese-en 確率的なユニタリ操作
 */
class QuantumGate_Probabilistic : public QuantumGateBase {
protected:
    Random random;
    std::vector<double> _distribution;
    std::vector<double> _cumulative_distribution;
    std::vector<QuantumGateBase*> _gate_list;

public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param distribution ゲートが現れる確率
     * @param gate_list ゲートのリスト
     */
    QuantumGate_Probabilistic(std::vector<double> distribution,
        std::vector<QuantumGateBase*> gate_list)
        : _distribution(distribution) {
        double sum = 0.;
        _cumulative_distribution.push_back(0.);
        for (auto val : distribution) {
            sum += val;
            _cumulative_distribution.push_back(sum);
        }
        std::transform(gate_list.cbegin(), gate_list.cend(),
            std::back_inserter(_gate_list),
            [](auto gate) { return gate->copy(); });
        FLAG_NOISE = true;
    };

    virtual ~QuantumGate_Probabilistic() {
        for (unsigned int i = 0; i < _gate_list.size(); ++i) {
            delete _gate_list[i];
        }
    }

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (state->is_state_vector()) {
            double r = random.uniform();
            auto ite = std::lower_bound(_cumulative_distribution.begin(),
                _cumulative_distribution.end(), r);
            assert(ite != _cumulative_distribution.begin());
            size_t gate_index =
                std::distance(_cumulative_distribution.begin(), ite) - 1;

            if (gate_index < _gate_list.size()) {
                _gate_list[gate_index]->update_quantum_state(state);
            }
        } else {
            auto org_state = state->copy();
            auto temp_state = state->copy();
            for (UINT gate_index = 0; gate_index < _gate_list.size();
                 ++gate_index) {
                if (gate_index == 0) {
                    _gate_list[gate_index]->update_quantum_state(state);
                    state->multiply_coef(_distribution[gate_index]);
                } else if (gate_index + 1 < _gate_list.size()) {
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
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override {
        std::vector<QuantumGateBase*> new_gate_list;
        std::transform(_gate_list.cbegin(), _gate_list.cend(),
            std::back_inserter(new_gate_list),
            [](auto gate) { return gate->copy(); });
        return new QuantumGate_Probabilistic(_distribution, new_gate_list);
    };

    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        std::cerr << "* Warning : Gate-matrix of probabilistic gate cannot be "
                     "obtained. Identity matrix is returned."
                  << std::endl;
        matrix = Eigen::MatrixXcd::Ones(1, 1);
    }

    /*
    added by kotamanegi.
    */

    virtual void set_seed(int seed) override { random.set_seed(seed); };

    virtual std::vector<double> get_cumulative_distribution() override {
        return _cumulative_distribution;
    };
    virtual std::vector<QuantumGateBase*> get_gate_list() override {
        return _gate_list;
    }
    virtual void optimize_ProbablisticGate() override {
        int n = _gate_list.size();
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
};

/**
 * \~japanese-en 選択内容を保存する確率的な操作
 */
class QuantumGate_ProbabilisticInstrument : public QuantumGateBase {
protected:
    Random random;
    std::vector<double> _distribution;
    std::vector<double> _cumulative_distribution;
    std::vector<QuantumGateBase*> _gate_list;
    UINT _classical_register_address;

public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param distribution ゲートが現れる確率
     * @param gate_list ゲートのリスト
     * @param classical_register 選択結果を保存するアドレス
     */
    QuantumGate_ProbabilisticInstrument(std::vector<double> distribution,
        std::vector<QuantumGateBase*> gate_list,
        UINT classical_register_address)
        : _distribution(distribution),
          _classical_register_address(classical_register_address) {
        double sum = 0.;
        _cumulative_distribution.push_back(0.);
        for (auto val : distribution) {
            sum += val;
            _cumulative_distribution.push_back(sum);
        }
        std::transform(gate_list.cbegin(), gate_list.cend(),
            std::back_inserter(_gate_list),
            [](auto gate) { return gate->copy(); });
    };

    virtual ~QuantumGate_ProbabilisticInstrument() {
        for (unsigned int i = 0; i < _gate_list.size(); ++i) {
            delete _gate_list[i];
        }
    }

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        double r = random.uniform();
        auto ite = std::lower_bound(_cumulative_distribution.begin(),
            _cumulative_distribution.end(), r);
        assert(ite != _cumulative_distribution.begin());
        size_t gate_index =
            std::distance(_cumulative_distribution.begin(), ite) - 1;

        if (gate_index < _gate_list.size()) {
            _gate_list[gate_index]->update_quantum_state(state);
        }
        state->set_classical_value(
            this->_classical_register_address, (UINT)gate_index);
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override {
        std::vector<QuantumGateBase*> new_gate_list;
        std::transform(_gate_list.cbegin(), _gate_list.cend(),
            std::back_inserter(new_gate_list),
            [](auto gate) { return gate->copy(); });
        return new QuantumGate_ProbabilisticInstrument(
            _distribution, new_gate_list, _classical_register_address);
    };

    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        std::cerr << "* Warning : Gate-matrix of probabilistic gate cannot be "
                     "obtained. Identity matrix is returned."
                  << std::endl;
        matrix = Eigen::MatrixXcd::Ones(1, 1);
    }
};

/**
 * \~japanese-en Kraus表現のCPTP-map
 */
class QuantumGate_CPTP : public QuantumGateBase {
protected:
    Random random;
    std::vector<QuantumGateBase*> _gate_list;

public:
    explicit QuantumGate_CPTP(std::vector<QuantumGateBase*> gate_list) {
        std::transform(gate_list.cbegin(), gate_list.cend(),
            std::back_inserter(_gate_list),
            [](auto gate) { return gate->copy(); });
    };
    virtual ~QuantumGate_CPTP() {
        for (unsigned int i = 0; i < _gate_list.size(); ++i) {
            delete _gate_list[i];
        }
    }

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (state->is_state_vector()) {
            double r = random.uniform();

            double sum = 0.;
            double org_norm = state->get_squared_norm();

            auto buffer = state->copy();
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
                }
            }
            if (!(r < sum)) {
                std::cerr << "* Warning : CPTP-map was not trace preserving. "
                             "Identity-map is applied."
                          << std::endl;
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

    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override {
        std::vector<QuantumGateBase*> new_gate_list;
        std::transform(_gate_list.cbegin(), _gate_list.cend(),
            std::back_inserter(new_gate_list),
            [](auto gate) { return gate->copy(); });
        return new QuantumGate_CPTP(new_gate_list);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        std::cerr << "* Warning : Gate-matrix of CPTP-map cannot be obtained. "
                     "Identity matrix is returned."
                  << std::endl;
        matrix = Eigen::MatrixXcd::Ones(1, 1);
    }
};

/**
 * \~japanese-en Kraus表現のCP-map
 */
class QuantumGate_CP : public QuantumGateBase {
protected:
    Random random;
    std::vector<QuantumGateBase*> _gate_list;
    const bool _state_normalize;
    const bool _probability_normalize;
    const bool _assign_zero_if_not_matched;

public:
    QuantumGate_CP(std::vector<QuantumGateBase*> gate_list,
        bool state_normalize, bool probability_normalize,
        bool assign_zero_if_not_matched)
        : _state_normalize(state_normalize),
          _probability_normalize(probability_normalize),
          _assign_zero_if_not_matched(assign_zero_if_not_matched) {
        std::transform(gate_list.cbegin(), gate_list.cend(),
            std::back_inserter(_gate_list),
            [](auto gate) { return gate->copy(); });
    };
    virtual ~QuantumGate_CP() {
        for (unsigned int i = 0; i < _gate_list.size(); ++i) {
            delete _gate_list[i];
        }
    }

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
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

    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override {
        std::vector<QuantumGateBase*> new_gate_list;
        std::transform(_gate_list.cbegin(), _gate_list.cend(),
            std::back_inserter(new_gate_list),
            [](auto gate) { return gate->copy(); });
        return new QuantumGate_CP(new_gate_list, _state_normalize,
            _probability_normalize, _assign_zero_if_not_matched);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        std::cerr << "* Warning : Gate-matrix of CPTP-map cannot be obtained. "
                     "Identity matrix is returned."
                  << std::endl;
        matrix = Eigen::MatrixXcd::Ones(1, 1);
    }
};

/**
 * \~japanese-en Instrument
 */
class QuantumGate_Instrument : public QuantumGateBase {
protected:
    Random random;
    std::vector<QuantumGateBase*> _gate_list;
    UINT _classical_register_address;

public:
    QuantumGate_Instrument(std::vector<QuantumGateBase*> gate_list,
        UINT classical_register_address)
        : _classical_register_address(classical_register_address) {
        std::transform(gate_list.cbegin(), gate_list.cend(),
            std::back_inserter(_gate_list),
            [](auto gate) { return gate->copy(); });
    };
    virtual ~QuantumGate_Instrument() {
        for (unsigned int i = 0; i < _gate_list.size(); ++i) {
            delete _gate_list[i];
        }
    }

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
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
            std::cerr << "* Warning : Instrument-map was not trace preserving. "
                         "Identity-map is applied."
                      << std::endl;
        }
        delete buffer;

        state->set_classical_value(_classical_register_address, index);
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override {
        std::vector<QuantumGateBase*> new_gate_list;
        std::transform(_gate_list.cbegin(), _gate_list.cend(),
            std::back_inserter(new_gate_list),
            [](auto gate) { return gate->copy(); });
        return new QuantumGate_Instrument(
            new_gate_list, _classical_register_address);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        std::cerr
            << "* Warning : Gate-matrix of Instrument cannot be obtained. "
               "Identity matrix is returned."
            << std::endl;
        matrix = Eigen::MatrixXcd::Ones(1, 1);
    }
};

/**
 * \~japanese-en Adaptiveな操作
 */
class QuantumGate_Adaptive : public QuantumGateBase {
protected:
    QuantumGateBase* _gate;
    std::function<bool(const std::vector<UINT>&)> _func_without_id;
    std::function<bool(const std::vector<UINT>&, UINT)> _func_with_id;
    const int _id;

public:
    QuantumGate_Adaptive(QuantumGateBase* gate,
        std::function<bool(const std::vector<UINT>&)> func_without_id)
        : _gate(gate->copy()), _func_without_id(func_without_id), _id(-1){};
    QuantumGate_Adaptive(QuantumGateBase* gate,
        std::function<bool(const std::vector<UINT>&, UINT)> func_with_id,
        UINT id)
        : _gate(gate->copy()),
          _func_with_id(func_with_id),
          _id(static_cast<int>(id)){};
    virtual ~QuantumGate_Adaptive() { delete _gate; }

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
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
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override {
        if (_id == -1) {
            return new QuantumGate_Adaptive(_gate->copy(), _func_without_id);
        } else {
            return new QuantumGate_Adaptive(_gate->copy(), _func_with_id, _id);
        }
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        std::cerr
            << "* Warning : Gate-matrix of Adaptive-gate cannot be obtained. "
               "Identity matrix is returned."
            << std::endl;
        matrix = Eigen::MatrixXcd::Ones(1, 1);
    }
};
