#pragma once

#include "gate.hpp"
#include "gate_merge.hpp"
#include "state.hpp"
#include "utility.hpp"
/**
 * ここら辺のtarget listの仕様について
 * ゲートをマージしたときのtargetやcontrolの挙動は
 * get_new_qubit_list 関数で決められている
 * Identity のゲート + 含まれるすべてのゲート
 * のゲート集合を元に、　get_new_qubit_list で決める
 * ただし、和が1のProbabilistic においてのみ、　Identityなしで求めている
 */

/**
 * \~japanese-en 確率的なユニタリ操作
 */
class QuantumGate_Probabilistic : public QuantumGateBase {
protected:
    Random random;
    std::vector<double> _distribution;
    std::vector<double> _cumulative_distribution;
    std::vector<QuantumGateBase*> _gate_list;
    bool is_instrument;
    UINT _classical_register_address;

public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param distribution ゲートが現れる確率
     * @param gate_list ゲートのリスト
     */
    explicit QuantumGate_Probabilistic(const std::vector<double>& distribution,
        const std::vector<QuantumGateBase*>& gate_list);

    explicit QuantumGate_Probabilistic(const std::vector<double>& distribution,
        const std::vector<QuantumGateBase*>& gate_list,
        UINT classical_register_address);

    virtual ~QuantumGate_Probabilistic();

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override;
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual QuantumGate_Probabilistic* copy() const override;

    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override;

    /**
     * \~japanese-en ptreeに変換する
     *
     * @return ptree
     */
    virtual boost::property_tree::ptree to_ptree() const override;

    /*
    added by kotamanegi.
    */

    virtual void set_seed(int seed) override;
    virtual std::vector<double> get_cumulative_distribution();
    virtual std::vector<double> get_distribution();
    virtual std::vector<QuantumGateBase*> get_gate_list();
    virtual void optimize_ProbablisticGate();
    virtual bool is_noise() override;
};

/**
 * \~japanese-en Kraus表現のCPTP-map
 */
class QuantumGate_CPTP : public QuantumGateBase {
protected:
    Random random;
    std::vector<QuantumGateBase*> _gate_list;
    bool is_instrument;
    UINT _classical_register_address;

public:
    explicit QuantumGate_CPTP(std::vector<QuantumGateBase*> gate_list);

    explicit QuantumGate_CPTP(std::vector<QuantumGateBase*> gate_list,
        UINT classical_register_address);

    virtual ~QuantumGate_CPTP();

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override;

    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual QuantumGate_CPTP* copy() const override;
    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override;

    /**
     * \~japanese-en ptreeに変換する
     *
     * @return ptree
     */
    virtual boost::property_tree::ptree to_ptree() const override;
    virtual std::vector<QuantumGateBase*> get_gate_list();
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
    explicit QuantumGate_CP(std::vector<QuantumGateBase*> gate_list,
        bool state_normalize, bool probability_normalize,
        bool assign_zero_if_not_matched);
    virtual ~QuantumGate_CP();

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override;

    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual QuantumGate_CP* copy() const override;
    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override;

    /**
     * \~japanese-en ptreeに変換する
     *
     * @return ptree
     */
    virtual boost::property_tree::ptree to_ptree() const override;
    virtual std::vector<QuantumGateBase*> get_gate_list();
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
    explicit QuantumGate_Adaptive(QuantumGateBase* gate,
        std::function<bool(const std::vector<UINT>&)> func_without_id);
    explicit QuantumGate_Adaptive(QuantumGateBase* gate,
        std::function<bool(const std::vector<UINT>&, UINT)> func_with_id,
        UINT id);
    virtual ~QuantumGate_Adaptive();

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override;

    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual QuantumGate_Adaptive* copy() const override;

    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override;
};

/**
 * This type alias is kept for backward compatibility.
 * Do not edit this!
 */
using QuantumGate_ProbabilisticInstrument = QuantumGate_Probabilistic;
using QuantumGate_Instrument = QuantumGate_CPTP;
