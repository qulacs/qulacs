/**
 * ゲートからゲートを生成する関数
 *
 * @file gate_merge.hpp
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
DllExport QuantumGateMatrix* merge(const QuantumGateBase* gate_applied_first,
    const QuantumGateBase* gate_applied_later);

/**
 * \~japanese-en 複数のゲートが連続して作用する新たなゲートを作成する。
 *
 * @param gate_list 作用するゲート列。先頭要素から順番に状態に作用する。
 * @return 合成したゲート
 */
DllExport QuantumGateMatrix* merge(std::vector<QuantumGateBase*> gate_list);

/**
 * \~japanese-en 二つのゲートのゲート行列を足して新たなゲートを作成する。
 *
 * TODO: control-qubitがあるときの挙動が未定義
 * @param gate1 先に状態に作用するゲート
 * @param gate2 後に状態に作用するゲート
 * @return 二つのゲートを足したゲート
 */
DllExport QuantumGateMatrix* add(
    const QuantumGateBase* gate1, const QuantumGateBase* gate2);

/**
 * \~japanese-en 複数のゲートを足して新たなゲートを作成する。
 *
 * TODO: control-qubitがあるときの挙動が未定義
 * @param gate_list 足すゲート列
 * @return 足したゲート
 */
DllExport QuantumGateMatrix* add(std::vector<QuantumGateBase*> gate_list);

/**
 * \~japanese-en
 * <code>QuantumGateBase</code>の任意のサブクラスを、<code>QuantumGateMatrix</code>のクラスの変換する。
 *
 * \f$n\f$-qubitゲートで\f$n\f$が非常に大きいとき、\f$d^2\f$の大量のメモリを使用する点に注意。
 * @param gate 変換するゲート
 * @return 変換された<code>QuantumGateMatrix</code>クラスのインスタンス
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
DllExport QuantumGateBase* Probabilistic(
    std::vector<double> distribution, std::vector<QuantumGateBase*> gate_list);

/**
 * \~japanese-en 密度行列にも確率的に作用する量子ゲートを作成する。
 *
 * 確率分布の総和が1でない場合、残った確率が採用されたときには何も作用しない。
 * @param distribution 確率分布
 * @param gate_list 作用する量子ゲート
 * @param classical_register_address 作用したゲートの添え字を保存するアドレス
 * @return 確率的に作用するゲート
 */
DllExport QuantumGateBase* ProbabilisticInstrument(
    std::vector<double> distribution, std::vector<QuantumGateBase*> gate_list,
    UINT classical_register_address);

/**
 * \~japanese-en CPTP-mapを作成する
 *
 * \f$p_i = {\rm Tr}[K_i \rho
 * K_i^{\dagger}]\f$を計算し、\f$\{p_i\}\f$の確率分布でクラウス演算子を採用して、\f$\sqrt{p_i}^{-1}\f$で正規化する。
 * @param gate_list クラウス演算を行うゲートのリスト
 * @return CPTP-map
 */
DllExport QuantumGateBase* CPTP(std::vector<QuantumGateBase*> gate_list);

/**
 * \~japanese-en CP-mapを作成する
 *
 * \f$p_i = {\rm Tr}[K_i \rho
 * K_i^{\dagger}]\f$を計算し、\f$\{p_i\}\f$の確率分布でクラウス演算子を採用する。
 * @param gate_list クラウス演算を行うゲートのリスト
 * @param state_normlize trueだったら状態を規格化する
 * @param probability_normalize trueだったら確率分布を規格化する
 * @param assign_zero_if_not_matched
 * どのKraus演算子にもマッチしなかったら0を代入する
 * @return CP-map
 */
DllExport QuantumGateBase* CP(std::vector<QuantumGateBase*> gate_list,
    bool state_normalize, bool probability_normalize,
    bool assign_zero_if_not_matched);

/**
 * \~japanese-en Instrumentを作成する
 *
 * InstrumentではCPTP-mapを作用させ、かつ作用されたクラウス演算子の添え字を<code>classical_register_address</code>に書き込む。
 * @param gate_list クラウス演算を行うゲートのリスト
 * @param classical_register_address 添え字を書きこむclassical registerの添え字
 * @return Instrument
 */
DllExport QuantumGateBase* Instrument(
    std::vector<QuantumGateBase*> gate_list, UINT classical_register_address);

/**
 * \~japanese-en 適応操作のゲートを作成する
 *
 * <code>func</code>が<code>true</code>を返すときのみ<code>gate</code>を作用する量子ゲートを作成する。
 * @param gate ゲート
 * @param func <code>std::vector<unsigned
 * int>&</code>を受け取り、<code>bool</code>を返す関数
 * @return Adaptive gate
 */
DllExport QuantumGateBase* Adaptive(
    QuantumGateBase* gate, std::function<bool(const std::vector<UINT>&)> func);
/**
 * \~japanese-en 適応操作のゲートを作成する
 *
 * <code>func</code>が<code>true</code>を返すときのみ<code>gate</code>を作用する量子ゲートを作成する。
 * @param gate ゲート
 * @param func <code>std::vector<unsigned
 * int>&,UINT</code>を受け取り、<code>bool</code>を返す関数
 * @param id funcに引数として与えるUINTの値
 * @return Adaptive gate
 */
DllExport QuantumGateBase* Adaptive(QuantumGateBase* gate,
    std::function<bool(const std::vector<UINT>&, UINT)> func, UINT id);

DllExport QuantumGateMatrix* get_transpose_gate(const QuantumGateBase* gate);

DllExport QuantumGateMatrix* get_conjugate_gate(const QuantumGateBase* gate);

DllExport QuantumGateMatrix* get_adjoint_gate(const QuantumGateBase* gate);

void get_new_qubit_list(const QuantumGateBase* gate_first,
    const QuantumGateBase* gate_second,
    std::vector<TargetQubitInfo>& new_target_list,
    std::vector<ControlQubitInfo>& new_control_list);
void get_extended_matrix(const QuantumGateBase* gate,
    const std::vector<TargetQubitInfo>& new_target_list,
    const std::vector<ControlQubitInfo>& new_control_list,
    ComplexMatrix& matrix);

}  // namespace gate
// namespace gate
