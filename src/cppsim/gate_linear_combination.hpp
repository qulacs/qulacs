#pragma once

#include "gate.hpp"
#include "state.hpp"
#include "utility.hpp"

/**
 * \~japanese-en いくつかのゲートの線型結合
 */
class QuantumGate_LinearCombination : public QuantumGateBase {
protected:
    std::vector<CPPCTYPE> _coefs;
    std::vector<QuantumGateBase*> _gate_list;

public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param coefs ゲートにかけられる係数
     * @param gate_list ゲートのリスト
     */
    explicit QuantumGate_LinearCombination(const std::vector<CPPCTYPE>& coefs,
        const std::vector<QuantumGateBase*>& gate_list)
        : _coefs(coefs) {
        if (coefs.size() != gate_list.size()) {
            throw InvalidCoefListException(
                "Error: "
                "QuantumGate_LinearCombination::LinearCombination(vector<"
                "CPPCTYPE>, vector<QuantumGateBase*>): gate_list.size() must "
                "be "
                "equal to coefs.size().");
        }
        _gate_list.reserve(gate_list.size());
        std::transform(gate_list.begin(), gate_list.end(),
            std::back_inserter(_gate_list),
            [&](const QuantumGateBase* gate) { return gate->copy(); });
    };

    virtual ~QuantumGate_LinearCombination() {
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
            auto* updated_state = state->copy();
            auto* tmp_state = state->copy();
            updated_state->set_zero_norm_state();
            for (UINT idx = 0; idx < _gate_list.size(); ++idx) {
                tmp_state->load(state);
                _gate_list[idx]->update_quantum_state(tmp_state);
                updated_state->add_state_with_coef(_coefs[idx], tmp_state);
            }
            state->load(updated_state);
            delete updated_state;
            delete tmp_state;
        } else {
            throw NotImplementedException(
                "QuantumGate_LinearCombination::update_quantum_state for "
                "density matrix is not supported.");
        }
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual QuantumGate_LinearCombination* copy() const override {
        return new QuantumGate_LinearCombination(_coefs, _gate_list);
    };

    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        std::cerr << "* Warning : Gate-matrix of linear combination gate is "
                     "currently not "
                     "supported. Identity matrix is returned."
                  << std::endl;
        matrix = Eigen::MatrixXcd::Ones(1, 1);
    }

    /**
     * \~japanese-en ptreeに変換する
     *
     * @return ptree
     */
    virtual boost::property_tree::ptree to_ptree() const override {
        boost::property_tree::ptree pt;
        pt.put("name", "LinearCombinationGate");
        boost::property_tree::ptree coefs_pt;
        for (CPPCTYPE c : _coefs) {
            boost::property_tree::ptree child;
            child.put("", c);
            coefs_pt.push_back(std::make_pair("", child));
        }
        pt.put_child("coefs", coefs_pt);
        boost::property_tree::ptree gate_list_pt;
        for (const QuantumGateBase* gate : _gate_list) {
            gate_list_pt.push_back(std::make_pair("", gate->to_ptree()));
        }
        pt.put_child("gate_list", gate_list_pt);
        return pt;
    }

    virtual std::vector<CPPCTYPE> get_coef_list() { return _coefs; };
    virtual std::vector<QuantumGateBase*> get_gate_list() { return _gate_list; }

    virtual bool is_noise() override { return true; }
};
