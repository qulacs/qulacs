/**
 * @file gate_merge.hpp
 *
 * \~japanese-en ゲートからゲートを生成する関数
 * \~english Functions to generate gate
 */
#pragma once

#include "gate.hpp"
#include "gate_matrix.hpp"

namespace gate {
    /**
     * \~japanese-en 二つのゲートが連続して作用する新たなゲートを作成する。
     *
     * @param gate_applied_first 先に状態に作用するゲート
     * @param gate_applied_later 後に状態に作用するゲート
     * @return 二つのゲートを合成したゲート
     */
    /**
     * \~english Create a new gate by two gates that work seccessively.
     *
     * @param gate_applied_first Gate operating first
     * @param gate_applied_later Gate operating later
     * @return A gate that combines two gates
     */
    DllExport QuantumGateMatrix* merge(const QuantumGateBase* gate_applied_first, const QuantumGateBase* gate_applied_later);

    /**
     * \~japanese-en 複数のゲートが連続して作用する新たなゲートを作成する。
     *
     * @param gate_list 作用するゲート列。先頭要素から順番に状態に作用する。
     * @return 合成したゲート
     */
    /**
     * \~english Create a new gate by several gates that work seccessively.
     *
     * @param gate_list The gate array to operate. Operate on states in order from the first element.
     * @return Created gate
     */
    DllExport QuantumGateMatrix* merge(std::vector<const QuantumGateBase*> gate_list);

    /**
     * \~japanese-en 二つのゲートのゲート行列を足して新たなゲートを作成する。
     *
     * TODO: control-qubitがあるときの挙動が未定義
     * @param gate1 先に状態に作用するゲート
     * @param gate2 後に状態に作用するゲート
     * @return 二つのゲートを足したゲート
     */
    /**
     * \~english Create a new gate by adding the gate matrices of two gates.
     *
     * TODO: Behavior undefined when there is control-qubit
     * @param gate1 Gate operating on state first
     * @param gate2 Gate operating on state later
     * @return Generated gate
     */
    DllExport QuantumGateMatrix* add(const QuantumGateBase* gate1, const QuantumGateBase* gate2);

    /**
     * \~japanese-en 複数のゲートを足して新たなゲートを作成する。
     *
     * TODO: control-qubitがあるときの挙動が未定義
     * @param gate_list 足すゲート列
     * @return 足したゲート
     */
    /**
     * \~english Create a new gate by adding multiple gates.
     *
     * TODO: Behavior undefined when there is control-qubit
     * @param gate_list Gate array to add
     * @return Generated
     */
    DllExport QuantumGateMatrix* add(std::vector<const QuantumGateBase*> gate_list);

    /**
     * \~japanese-en <code>QuantumGateBase</code>の任意のサブクラスを、<code>QuantumGateMatrix</code>のクラスの変換する。
     *
     * \f$n\f$-qubitゲートで\f$n\f$が非常に大きいとき、\f$d^2\f$の大量のメモリを使用する点に注意。
     * @param gate 変換するゲート
     * @return 変換された<code>QuantumGateMatrix</code>クラスのインスタンス
     */
    /**
     * \~english Convert any subclass of <code>QuantumGateBase</code> into class of <code>QuantumGateMatrix</code>.
     *
     * Note that the \f$n\f$-qubit gate uses a large amount memory of \f$d^2\f$ when \f$n\f$ is very large.
     * @param gate Gate to convert
     * @return Instance of the converted <code>QuantumGateMatrix</code> class
     */
    DllExport QuantumGateMatrix* to_matrix_gate(const QuantumGateBase* gate);

    /**
     * \~japanese-en 確率的に作用する量子ゲートを作成する。
     *
     * 確率分布の総和が1でない場合、残った確率が採用されたときには何も作用しない。
     * @param distribution 確率分布
     * @param gate_list 作用する量子ゲート
     * @return 確率的に作用するゲート
     */
    /**
     * \~english Create a quantum gate that works stochastically.
     *
     * If the sum of the probability distributions is not 1, nothing happens when the remaining probabilities are adopted.
     * @param distribution Probability distribution
     * @param gate_list Working quantum gate
     * @return Stochastic gates
     */
    DllExport QuantumGateBase* Probabilistic(std::vector<double> distribution, std::vector<QuantumGateBase*> gate_list);

    /**
     * \~japanese-en CPTP-mapを作成する
     *
     * \f$p_i = {\rm Tr}[K_i \rho K_i^{\dagger}]\f$を計算し、\f$\{p_i\}\f$の確率分布でクラウス演算子を採用して、\f$\sqrt{p_i}^{-1}\f$で正規化する。
     * @param gate_list クラウス演算を行うゲートのリスト
     * @return CPTP-map
     */
    /**
     * \~english Create CPTP-map
     *
     * Calculate \f$p_i = {\rm Tr}[K_i \rho K_i^{\dagger}]\f$, adopt the Claus operator in the probability distribution of \f$\{p_i\}\f$, and normalize with \f$\sqrt{p_i}^{-1}\f$.
     * @param gate_list List of gates to perform Claus operation
     * @return CPTP-map
     */
    DllExport QuantumGateBase* CPTP(std::vector<QuantumGateBase*> gate_list);

    /**
     * \~japanese-en Instrumentを作成する
     *
     * InstrumentではCPTP-mapを作用させ、かつ作用されたクラウス演算子の添え字を<code>classical_register_address</code>に書き込む。
     * @param gate_list クラウス演算を行うゲートのリスト
     * @param classical_register_address 添え字を書きこむclassical registerの添え字
     * @return Instrument
     */
    /**
     * \~english Create instrument
     *
     * The Instrument operates the CPTP-map and writes the index of the operated Claus operator to <code>classical_register_address</code>.
     * @param gate_list List of gates to perform Claus operation
     * @param classical_register_address Subscript of classical register to write subscript
     * @return Instrument
     */
    DllExport QuantumGateBase* Instrument(std::vector<QuantumGateBase*> gate_list, UINT classical_register_address);

    /**
     * \~japanese-en 適応操作のゲートを作成する
     *
     * <code>func</code>が<code>true</code>を返すときのみ<code>gate</code>を作用する量子ゲートを作成する。
     * @param gate ゲート
     * @param func <code>std::vector<unsigned int>&</code>を受け取り、<code>bool</code>を返す関数
     * @return Adaptive gate
     */
    /**
     * \~english Create an adaptive operation gate
     *
     * Create a quantum gate that operates on <code>gate</code> when <code>func</code> returns <code>true</code>.
     * @param gate Gate
     * @param func Functin that receives <code>std::vector<unsigned int>&</code> and returns <code>bool</code>.
     * @return Adaptive gate
     */
    DllExport QuantumGateBase* Adaptive(QuantumGateBase* gate, std::function<bool(const std::vector<UINT>&)> func);
}
