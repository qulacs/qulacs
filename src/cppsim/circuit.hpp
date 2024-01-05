
/**
 * @file circuit.hpp
 *
 * @brief QuantumCircuitクラスの詳細
 */

#pragma once

#include <string>
#include <vector>

#include "exception.hpp"
#include "type.hpp"
#include "utility.hpp"

class QuantumStateBase;
class QuantumGateBase;
class PauliOperator;
class HermitianQuantumOperator;
using Observable = HermitianQuantumOperator;

/**
 * \~japanese-en 量子回路のクラス
 *
 * 量子回路を管理するクラス。QuantumGateクラスをリストとして持ち、種々の操作を行う。
 * 管理する量子ゲートは量子回路の解放時にすべて解放される。
 */
class DllExport QuantumCircuit {
protected:
    std::vector<QuantumGateBase*> _gate_list;
    UINT _qubit_count;

    // prohibit shallow copy
    QuantumCircuit(const QuantumCircuit& obj);
    QuantumCircuit& operator=(const QuantumCircuit&) = delete;

public:
    const UINT& qubit_count; /**< \~japanese-en 量子ビットの数*/
    const std::vector<QuantumGateBase*>&
        gate_list; /**< \~japanese-en 量子ゲートのリスト*/

    /**
     * \~japanese-en 空の量子回路を作成する
     *
     * @param[in] qubit_count 量子ビットの数
     * @return 生成された量子回路のインスタンス
     */
    explicit QuantumCircuit(UINT qubit_count);

    /**
     * \~japanese-en 量子回路のディープコピーを生成する
     *
     * @return 量子回路のディープコピー
     */
    QuantumCircuit* copy() const;

    /**
     * \~japanese-en デストラクタ
     */
    virtual ~QuantumCircuit();

    ////////////////////// BASIC CONTROL OF QUANTUM CIRCUIT

    /**
     * \~japanese-en 量子ゲートを回路の末尾に追加する
     *
     * 量子ゲートを回路に追加する。
     * 追加した量子ゲートは量子回路の解放時に開放される。
     * @param[in] gate 追加する量子ゲート
     */
    virtual void add_gate(QuantumGateBase* gate);

    /**
     * \~japanese-en 量子ゲートを回路の指定位置に追加する。
     *
     * 量子ゲートを回路の指定した位置に追加する。
     * 追加した量子ゲートは量子回路の解放時に開放される。
     * @param[in] gate 追加する量子ゲート
     * @param[in] index 追加する位置
     */
    virtual void add_gate(QuantumGateBase* gate, UINT index);

    /**
     * \~japanese-en 量子ゲートを回路の末尾に追加する
     *
     * 与えられた量子ゲートのコピーを回路に追加する。
     * add_gateに比べコピーが発生する分低速な一方、引数で与えたゲートを再利用できる。
     * @param[in] gate 追加する量子ゲート
     */
    virtual void add_gate_copy(const QuantumGateBase* gate);

    /**
     * \~japanese-en 量子ゲートを回路の指定位置に追加する。
     *
     * 与えらた量子ゲートを回路の指定した位置に追加する。
     * @param[in] gate 追加する量子ゲート
     * @param[in] index 追加する位置
     */
    virtual void add_gate_copy(const QuantumGateBase* gate, UINT index);

    /**
     * \~japanese-en ノイズ付き量子ゲートを回路の末尾に追加する
     *
     * ノイズ付き量子ゲートを回路に追加する。
     * 追加したノイズ付き量子ゲートは量子回路の解放時に開放される。
     * @param[in] gate 追加するノイズ付き量子ゲート
     * @param[in] noise_type 追加するノイズの種類
     * @param[in] noise_prob ノイズが発生する確率
     */
    virtual void add_noise_gate(
        QuantumGateBase* gate, std::string noise_type, double noise_prob);
    /**
     * \~japanese-en ノイズ付き量子ゲートのコピーを回路の末尾に追加する
     *
     * ノイズ付き量子ゲートのコピーを回路に追加する。
     * add_noise_gateに比べコピーが発生する分低速な一方、引数で与えたゲートを再利用できる。
     * @param[in] gate 追加するノイズ付き量子ゲート
     * @param[in] noise_type 追加するノイズの種類
     * @param[in] noise_prob ノイズが発生する確率
     */
    void add_noise_gate_copy(
        QuantumGateBase* gate, std::string noise_type, double noise_prob);
    /**
     * \~japanese-en 量子回路からゲートを削除する。
     *
     * 削除した量子ゲートは解放される。
     * @param[in] index 削除するゲートの位置
     */
    virtual void remove_gate(UINT index);
    /**
     * \~japanese-en 量子回路内のゲートを移動する。
     *
     * 回路内の量子ゲートを移動する。
     * from_indexとto_indexの間のゲートはfrom_indexの方向にシフトする。
     * @param[in] from_index 移動元のゲートの位置
     * @param[in] to_index 移動先のゲートの位置
     */
    virtual void move_gate(UINT from_index, UINT to_index);
    /**
     *  \~japanese-en 量子回路をマージする。
     *
     * 引数で与えた量子回路のゲートを後ろに追加していく。
     * マージされた側の量子回路に変更を加えてもマージした側の量子回路には変更は加わらないことに注意する。
     * circuit1.add_circuit(circuit2)
     * circuit2.add_gate(gate) # これをしても、circuit1にgateは追加されない
     *
     * @param[in] circuit マージする量子回路
     */
    void merge_circuit(const QuantumCircuit* circuit) {
        for (auto gate : circuit->gate_list) {
            this->add_gate_copy(gate);
        }
        return;
    }
    /////////////////////////////// UPDATE QUANTUM STATE

    /**
     * \~japanese-en 量子状態を更新する
     *
     * 順番にすべての量子ゲートを作用する。量子状態の初期化などは行わない。
     * @param[in,out] state 作用する量子状態
     */
    void update_quantum_state(QuantumStateBase* state);

    /**
     * \~japanese-en 量子回路の指定範囲のみを用いて量子状態をを更新する
     *
     * 添え字がstart_indexからend_index-1までの量子ゲートを順番に量子ゲートを作用する。量子状態の初期化などは行わない。
     * @param[in,out] state 作用する量子状態
     * @param[in] start_index 開始位置
     * @param[in] end_index 修了位置
     */
    void update_quantum_state(
        QuantumStateBase* state, UINT start_index, UINT end_index);

    /**
     * \~japanese-en 量子状態を更新する(random seed指定)
     *
     * 順番にすべての量子ゲートを作用する。量子状態の初期化などは行わない。
     * @param[in,out] state 作用する量子状態
     * @param[in] seed 乱数の種
     */
    void update_quantum_state(QuantumStateBase* state, UINT seed);

    /**
     * \~japanese-en 量子回路の指定範囲のみを用いて量子状態をを更新する(random
     * seed指定)
     *
     * 添え字がstart_indexからend_index-1までの量子ゲートを順番に量子ゲートを作用する。量子状態の初期化などは行わない。
     * @param[in,out] state 作用する量子状態
     * @param[in] start_index 開始位置
     * @param[in] end_index 修了位置
     * @param[in] seed 乱数の種
     */
    void update_quantum_state(
        QuantumStateBase* state, UINT start_index, UINT end_index, UINT seed);

    /////////////////////////////// CHECK PROPERTY OF QUANTUM CIRCUIT

    /**
     * \~japanese-en 量子回路がCliffordかどうかを判定する。
     *
     * 全ての量子ゲートがCliffordである場合にtrueと判定される。
     * Non-Cliffordゲートが複数あり積がCliffordとなっている場合もfalseとして判定される点に注意。
     * @retval true Clifford
     * @retval false Non-Clifford
     */
    bool is_Clifford() const;

    /**
     * \~japanese-en 量子回路がFermionic Gaussianかどうかを判定する。
     *
     * 全ての量子ゲートがFermionic Gaussianである場合にtrueと判定される。
     * Non-Gaussianゲートが複数あり、結果としてGaussianとなっている場合もNon-Gaussianとして判定される点に注意。
     * @retval true Fermionic Gaussian
     * @retval false Non-fermionic Gaussian
     */
    bool is_Gaussian() const;

    /**
     * \~japanese-en 量子回路のdepthを計算する。
     *
     * ここでいうdepthとは、可能な限り量子ゲートを並列実行した時に掛かるステップ数を指す。
     * @return 量子回路のdepth
     */
    UINT calculate_depth() const;

    /**
     * \~japanese-en 量子回路のデバッグ情報の文字列を生成する
     *
     * @return 生成した文字列
     */
    virtual std::string to_string() const;

    /**
     * \~japanese-en 量子回路のデバッグ情報を出力する。
     *
     * @return 受け取ったストリーム
     */
    friend DllExport std::ostream& operator<<(
        std::ostream& os, const QuantumCircuit&);

    /**
     * \~japanese-en 量子回路のデバッグ情報を出力する。
     *
     * @return 受け取ったストリーム
     */
    friend DllExport std::ostream& operator<<(
        std::ostream& os, const QuantumCircuit* gate);

    /**
     * \~japanese-en \f$X\f$ gateを追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_X_gate(UINT target_index);

    /**
     * \~japanese-en \f$Y\f$ gateを追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_Y_gate(UINT target_index);

    /**
     * \~japanese-en \f$Z\f$ gateを追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_Z_gate(UINT target_index);

    /**
     * \~japanese-en Hadamard gateを追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_H_gate(UINT target_index);

    /**
     * \~japanese-en \f$S\f$ gateを追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_S_gate(UINT target_index);

    /**
     * \~japanese-en \f$S^{\dagger}\f$ gateを追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_Sdag_gate(UINT target_index);

    /**
     * \~japanese-en \f$T\f$ gateを追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_T_gate(UINT target_index);

    /**
     * \~japanese-en \f$T^{\dagger}\f$ gateを追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_Tdag_gate(UINT target_index);

    /**
     * \~japanese-en \f$\sqrt{X}\f$ gateを追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_sqrtX_gate(UINT target_index);

    /**
     * \~japanese-en \f$\sqrt{X}^{\dagger}\f$ gateを追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_sqrtXdag_gate(UINT target_index);

    /**
     * \~japanese-en \f$\sqrt{Y}\f$ gateを追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_sqrtY_gate(UINT target_index);

    /**
     * \~japanese-en \f$\sqrt{Y}^{\dagger}\f$ gateを追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_sqrtYdag_gate(UINT target_index);

    /**
     * \~japanese-en 0状態への射影演算を追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_P0_gate(UINT target_index);

    /**
     * \~japanese-en 1状態への射影演算を追加する。
     *
     * @param[in] target_index 作用する量子ビットの添え字
     */
    virtual void add_P1_gate(UINT target_index);

    /**
     * \~japanese-en CNOT gateを追加する。
     *
     * @param[in] control_index 作用するcontrol qubitの添え字
     * @param[in] target_index 作用するtarget qubitの添え字
     */
    virtual void add_CNOT_gate(UINT control_index, UINT target_index);

    /**
     * \~japanese-en Control-Z gateを追加する。
     *
     * @param[in] control_index 作用するcontrol qubitの添え字
     * @param[in] target_index 作用するtarget qubitの添え字
     */
    virtual void add_CZ_gate(UINT control_index, UINT target_index);

    /**
     * \~japanese-en SWAP gateを追加する。
     *
     * @param[in] target_index1 作用するtarget qubitの添え字
     * @param[in] target_index2 作用するもう一方のtarget qubitの添え字
     */
    virtual void add_SWAP_gate(UINT target_index1, UINT target_index2);

    /**
     * \~japanese-en FusedSWAP gateを追加する。
     *
     * @param[in] target_index1 作用するqubitブロックの先頭の添え字
     * @param[in] target_index2 作用するもう一方のqubitブロックの先頭の添え字
     * @param[in] block_size 作用するqubitブロックの大きさ
     */
    virtual void add_FusedSWAP_gate(
        UINT target_index1, UINT target_index2, UINT block_size);

    /**
     * \~japanese-en X-rotation gateを追加する。
     *
     * ゲートの表記は
     *
     * @f[
     * R_X(\theta) = \exp(i\frac{\theta}{2} X) =
     *     \begin{pmatrix}
     *     \cos(\frac{\theta}{2})  & i\sin(\frac{\theta}{2}) \\
     *     i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
     *     \end{pmatrix}
     * @f]
     *
     * である。
     *
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] angle 回転角\f$\theta\f$
     */
    virtual void add_RX_gate(UINT target_index, double angle);

    /**
     * \~japanese-en Y-rotation gateを追加する。
     *
     * ゲートの表記は
     *
     * @f[
     * R_Y(\theta) = \exp(i\frac{\theta}{2} Y) =
     *     \begin{pmatrix}
     *     \cos(\frac{\theta}{2})  & \sin(\frac{\theta}{2}) \\
     *     -\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
     *     \end{pmatrix}
     * @f]
     *
     * である。
     *
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] angle 回転角\f$\theta\f$
     */
    virtual void add_RY_gate(UINT target_index, double angle);

    /**
     * \~japanese-en Z-rotation gateを追加する。
     *
     * ゲートの表記は
     *
     * @f[
     * R_Z(\theta) = \exp(i\frac{\theta}{2} Z) =
     *     \begin{pmatrix}
     *     e^{i\frac{\theta}{2}} & 0 \\
     *     0 & e^{-i\frac{\theta}{2}}
     *     \end{pmatrix}
     * @f]
     *
     * である。
     *
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] angle 回転角\f$\theta\f$
     */
    virtual void add_RZ_gate(UINT target_index, double angle);

    /**
     * \~japanese-en X-rotation gateを追加する。
     *
     * ゲートの表記は
     *
     * @f[
     * R_X(\theta) = \exp(i\frac{\theta}{2} X) =
     *     \begin{pmatrix}
     *     \cos(\frac{\theta}{2})  & i\sin(\frac{\theta}{2}) \\
     *     i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
     *     \end{pmatrix}
     * @f]
     *
     * である。
     * 一般的な表記に対して逆向き,qulacsのRXと同じ向きである。
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] angle 回転角\f$\theta\f$
     */
    virtual void add_RotInvX_gate(UINT target_index, double angle);

    /**
     * \~japanese-en Y-rotation gateを追加する。
     *
     * ゲートの表記は
     *
     * @f[
     * R_Y(\theta) = \exp(i\frac{\theta}{2} Y) =
     *     \begin{pmatrix}
     *     \cos(\frac{\theta}{2})  & \sin(\frac{\theta}{2}) \\
     *     -\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
     *     \end{pmatrix}
     * @f]
     *
     * である。
     * 一般的な表記に対して逆向き,qulacsのRYと同じ向きである。
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] angle 回転角\f$\theta\f$
     */
    virtual void add_RotInvY_gate(UINT target_index, double angle);

    /**
     * \~japanese-en Z-rotation gateを追加する。
     *
     * ゲートの表記は
     *
     * @f[
     * R_Z(\theta) = \exp(i\frac{\theta}{2} Z) =
     *     \begin{pmatrix}
     *     e^{i\frac{\theta}{2}} & 0 \\
     *     0 & e^{-i\frac{\theta}{2}}
     *     \end{pmatrix}
     * @f]
     *
     * である。
     * 一般的な表記に対して逆向き,qulacsのRZと同じ向きである。
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] angle 回転角\f$\theta\f$
     */
    virtual void add_RotInvZ_gate(UINT target_index, double angle);

    /**
     * \~japanese-en X-rotation gateを追加する。
     *
     * ゲートの表記は
     *
     * @f[
     * RotX(\theta) = \exp(-i\frac{\theta}{2} X) =
     *     \begin{pmatrix}
     *     \cos(\frac{\theta}{2})  & -i\sin(\frac{\theta}{2}) \\
     *     -i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
     *     \end{pmatrix}
     * @f]
     *
     * である。
     * 一般的な表記と同じ向き,qulacsのRXに対して逆向きである。
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] angle 回転角\f$\theta\f$
     */
    virtual void add_RotX_gate(UINT target_index, double angle);

    /**
     * \~japanese-en Y-rotation gateを追加する。
     *
     * ゲートの表記は
     *
     * @f[
     * RotY(\theta) = \exp(-i\frac{\theta}{2} Y) =
     *     \begin{pmatrix}
     *     \cos(\frac{\theta}{2})  & -\sin(\frac{\theta}{2}) \\
     *     \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
     *     \end{pmatrix}
     * @f]
     *
     * である。
     * 一般的な表記と同じ向き,qulacsのRYに対して逆向きである。
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] angle 回転角\f$\theta\f$
     */
    virtual void add_RotY_gate(UINT target_index, double angle);

    /**
     * \~japanese-en Z-rotation gateを追加する。
     *
     * ゲートの表記は
     *
     * @f[
     * RotZ(\theta) = \exp(-i\frac{\theta}{2} Z) =
     *     \begin{pmatrix}
     *     e^{-i\frac{\theta}{2}} & 0 \\
     *     0 & e^{i\frac{\theta}{2}}
     *     \end{pmatrix}
     * @f]
     *
     * である。
     *
     * 一般的な表記と同じ向き,qulacsのRZに対して逆向きである。
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] angle 回転角\f$\theta\f$
     */
    virtual void add_RotZ_gate(UINT target_index, double angle);

    /**
     * \~japanese-en OpenQASMのu1 gateを追加する。
     *
     * ゲートの表記はIBMQのページを参照。
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] phi 回転角\f$\phi\f$
     */
    virtual void add_U1_gate(UINT target_index, double phi);

    /**
     * \~japanese-en OpenQASMのu2 gateを追加する。
     *
     * ゲートの表記はIBMQのページを参照。
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] phi 回転角\f$\phi\f$
     * @param[in] psi 回転角\f$\psi\f$
     */
    virtual void add_U2_gate(UINT target_index, double phi, double psi);

    /**
     * \~japanese-en OpenQASMのu3 gateを追加する。
     *
     * ゲートの表記はIBMQのページを参照。
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] phi 回転角\f$\phi\f$
     * @param[in] psi 回転角\f$\psi\f$
     * @param[in] lambda 回転角\f$\lambda\f$
     */
    virtual void add_U3_gate(
        UINT target_index, double phi, double psi, double lambda);

    /**
     * \~japanese-en n-qubitパウリゲートを追加する。
     *
     * n-qubitパウリゲートを作用する。
     * パウリ演算子は\f$(I,X,Y,Z) \mapsto (0,1,2,3)\f$と対応している。
     * 例えば、\f$X_3 Y_2 Z_4\f$であれば、target_index_list = {3,2,4},
     * pauli_id_list = {1,2,3}である。
     * 1-qubitパウリゲートとn-qubitパウリゲートの計算コストはほぼ同じため、パウリゲートのテンソル積を作用する場合はパウリゲートとして作用した方が処理が高速になる。
     * @param[in] target_index_list 作用するtarget qubitの添え字のリスト
     * @param[in] pauli_id_list target_index_listに対応したパウリ演算子のid
     */
    virtual void add_multi_Pauli_gate(
        std::vector<UINT> target_index_list, std::vector<UINT> pauli_id_list);

    /**
     * \~japanese-en n-qubitパウリゲートを追加する。
     *
     * n-qubitパウリゲートを作用する。
     * @param[in] pauli_operator 追加するパウリ演算子
     */
    virtual void add_multi_Pauli_gate(const PauliOperator& pauli_operator);

    /**
     * \~japanese-en n-qubitパウリ回転ゲートを追加する。
     *
     * n-qubitパウリ回転ゲートを作用する。
     * パウリ演算子は{I,X,Y,Z} = {0,1,2,3}と対応している。
     * 例えば、\f$\exp(i\theta X_3 Y_2 Z_4)\f$であれば、target_index_list =
     * {3,2,4}, pauli_id_list = {1,2,3}, angle = \f$\theta\f$とする。
     * 1-qubitパウリゲートとn-qubitパウリゲートの計算コストはほぼ同じため、パウリゲートのテンソル積を作用する場合はパウリゲートとして作用した方が処理が高速になる。
     * @param[in] target_index_list 作用するtarget qubitの添え字のリスト
     * @param[in] pauli_id_list target_index_listに対応したパウリ演算子のid
     * @param[in] angle 回転角
     */
    virtual void add_multi_Pauli_rotation_gate(
        std::vector<UINT> target_index_list, std::vector<UINT> pauli_id_list,
        double angle);

    /**
     *
     * \~japanese-en n-qubitパウリ回転ゲートを追加する。
     * n-qubitパウリ回転ゲートを作用する。
     * 回転角はPauliOperatorの係数を用いる。
     * @param[in] pauli_operator 追加するパウリ演算子
     */
    virtual void add_multi_Pauli_rotation_gate(
        const PauliOperator& pauli_operator);

    /**
     * \~japanese-en n-qubitオブザーバブル回転ゲートを追加する。(対角のみ)
     *
     * n-qubitオブザーバブル回転ゲートを作用する。ここで用いるオブザーバブルは、対角である必要がある。
     * @param[in] observable 追加するオブザーバブル
     * @param[in] angle 回転角
     */
    virtual void add_diagonal_observable_rotation_gate(
        const Observable& observable, double angle);

    /**
     * \~japanese-en n-qubitオブザーバブル回転ゲートを追加する。
     *
     * Suzuki-Trotter展開によりn-qubitオブザーバブル回転ゲートを作用する。ここで用いるオブザーバブルは、対角でなくてもよい。
     * @param[in] observable 追加するオブザーバブル
     * @param[in] angle 回転角 \f$ \theta \f$
     * @param[in] num_repeats
     * Trotter展開をする際の分割数\f$N\f$。指定しない場合は、関数内部で\f$
     * \#qubit \cdot \theta / N = 0.01\f$ となるように設定される。
     */
    virtual void add_observable_rotation_gate(
        const Observable& observable, double angle, UINT num_repeats = 0);
    /**
     * \~japanese-en 1-qubitのdenseな行列のゲートを追加する。
     *
     * denseな行列はユニタリである必要はなく、射影演算子やクラウス演算子でも良い。
     * @param[in] target_index 作用するtarget qubitの添え字
     * @param[in] matrix 作用する2*2の行列
     */
    virtual void add_dense_matrix_gate(
        UINT target_index, const ComplexMatrix& matrix);

    /**
     * \~japanese-en multi qubitのdenseな行列のゲートを追加する。
     *
     * denseな行列はユニタリである必要はなく、射影演算子やクラウス演算子でも良い。
     * matrixの次元はtarget_index_listのサイズを\f$m\f$としたとき\f$2^m\f$でなければいけない。
     * @param[in] target_index_list 作用するtarget qubitの添え字のリスト
     * @param[in] matrix 作用する行列
     */
    virtual void add_dense_matrix_gate(
        std::vector<UINT> target_index_list, const ComplexMatrix& matrix);

    /**
     * \~japanese-en multi qubitのランダムユニタリゲートを追加する。
     *
     * @param[in] target_index_list 作用するtarget qubitの添え字のリスト
     * @param[in] seed 乱数のseed値
     */
    virtual void add_random_unitary_gate(std::vector<UINT> target_index_list);
    virtual void add_random_unitary_gate(
        std::vector<UINT> target_index_list, UINT seed);

    /**
     * \~japanese-en ptreeに変換
     *
     * @return ptree
     */
    virtual boost::property_tree::ptree to_ptree() const;

    virtual QuantumCircuit* get_inverse(void);
};

namespace circuit {
/**
 * \~japanese-en ptreeからQuantumCircuitを構築する
 */
QuantumCircuit* from_ptree(const boost::property_tree::ptree& pt);
}  // namespace circuit
