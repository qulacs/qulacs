
/**
 * @file gate.hpp
 * @brief Definition and basic functions for QuantumGate
 */

/**
 * 
 * Regulation for argument/qubit/index ordering
 * Read this before treating kronecker product
 * 
 * --- arguments ---
 * Arguments of create_gate_*** function must be ordered as [control qubit (list), target qubit (list), rotation angle, state, dimension]
 * 
 * E.g. arguments of controlled gate must be ordered as [control qubit, target qubit].
 * CNOT_gate(0,1) = gate_from_string("CNOT 0 1") = control-0 X-1 gate
 *
 * When we perform gate->add_control(i, value), the operation is performed on the index where its i-th bit is value, i.e. if (index&(1ULL<<i) != 0)
 *  
 * --- state ---
 * In C/C++, it is useful to order the computational basis so that the RIGHTmost bit is the lowest bit.
 * e.g. state = [ state[0b00], state[0b01], state[0b10], state[0b11] ]
 * 
 * In quantum circuit, we call the TOP-line qubit as the FIRST qubit.
 *
 * In the braket representation, we consider the RIGHTmost bit in ket represents the FIRST (or TOP-line) qubit in the quantum circuit.
 * state[0b01] = <01|state|01> = the probability with which the TOP-qubit is one, and the second-top qubit is zero.
 * 
 * 
 * --- gate ---
 * To be consistent with the above index ordering, the order of tensor product is *** REVERSED ***.
 * 
 * X_0 has matrix representation : 
 * X_0 = I \\otimes X = np.kron(I,X) = 
 * [0,1,0,0] |00> state[00]
 * [1,0,0,0] |01> state[01]
 * [0,0,0,1] |10> state[10]
 * [0,0,1,0] |11> state[11]
 * 
 * CNOT(0,1) = control-0 target-NOT-1 has matrix representation
 * [1,0,0,0] |00>
 * [0,0,0,1] |01>
 * [0,0,1,0] |10>
 * [0,1,0,0] |11>
 *
 * X_0 Y_1 Z_2 = np.kron(Z_2, np.kron(Y1, X0))
 * 
 */


#pragma once

#include "type.hpp"
#include "qubit_info.hpp"
#include <vector>
#include <iostream>
#include <ostream>
#include <string>

 //! Flgas for gate property: gate is Pauli
#define FLAG_PAULI 0x01
 //! Flgas for gate property: gate is Clifford
#define FLAG_CLIFFORD 0x02
 //! Flgas for gate property: gate is Gaussian
#define FLAG_GAUSSIAN 0x04
 //! Flgas for gate property: gate is parametrized
#define FLAG_PARAMETRIC 0x08

class QuantumGateMatrix;
class QuantumGateSparseMatrix;
class QuantumStateBase;

/**
 * \~japanese-en 量子ゲートの基底クラス
 */
class DllExport QuantumGateBase {
protected:
    std::vector<TargetQubitInfo> _target_qubit_list; 
    std::vector<ControlQubitInfo> _control_qubit_list;
    UINT _gate_property=0;                            /**< \~japanese-en property of whole gate (e.g. Clifford or Pauli)*/
    std::string _name="Generic gate";

    // prohibit constructor, destructor, copy constructor, and insertion
    QuantumGateBase():
		target_qubit_list(_target_qubit_list), control_qubit_list(_control_qubit_list) 
	{};
	QuantumGateBase(const QuantumGateBase& obj):
		target_qubit_list(_target_qubit_list), control_qubit_list(_control_qubit_list)
	{
		_gate_property = obj._gate_property;
		_name = obj._name;
		_target_qubit_list = obj.target_qubit_list;
		_control_qubit_list = obj.control_qubit_list;
	};
    QuantumGateBase& operator=(const QuantumGateBase& rhs) = delete;
public:
    /**
     * \~japanese-en デストラクタ
     */
    virtual ~QuantumGateBase() {};

	const std::vector<TargetQubitInfo>& target_qubit_list; /**< ターゲット量子ビットのリスト */
	const std::vector<ControlQubitInfo>& control_qubit_list; /**< コントロール量子ビットのリスト */

    /**
     * \~japanese-en ターゲット量子ビットの添え字のリストを取得する
     * 
     * @return 量子ビットの添え字のリスト
     */
    std::vector<UINT> get_target_index_list() const {
        std::vector<UINT> res(target_qubit_list.size());
        for (UINT i = 0; i < target_qubit_list.size(); ++i) res[i] = target_qubit_list[i].index();
        return res;
    }
    /**
     * \~japanese-en コントロール量子ビットの添え字のリストを取得する
     * 
     * @return 量子ビットの添え字のリスト
     */
    std::vector<UINT> get_control_index_list() const {
        std::vector<UINT> res(control_qubit_list.size());
        for (UINT i = 0; i < control_qubit_list.size(); ++i) res[i] = control_qubit_list[i].index();
        return res;
    }
    /**
     * \~japanese-en 量子ビットの回転角を取得する
     *
     * @return 量子ビットの回転角
     */
    virtual double get_angle() const {
        return 0.0;
    }

    /**
     * \~japanese-en 量子状態を更新する
     * 
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state)  = 0 ;
    /**
     * \~japanese-en 自身のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const = 0;
    /**
     * \~japanese-en 自身のゲート行列をセットする
     * 
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const = 0;

    /**
     * \~japanese-en 与えれたゲート<code>gate</code>と自身が可換かを判定する
     * 
     * 非可換であると判定されても、実際には可換であることもある。
     * 可換と判定された場合は必ず可換である。
     * @param gate 可換か比較するゲート
     * @return true 可換である
     * @return false 非可換である
     */
    bool is_commute(const QuantumGateBase* gate) const;
    /**
     * \~japanese-en ゲートがパウリゲートかどうかを判定する
     * 
     * @return true パウリゲートである
     * @return false パウリゲートではない
     */
    bool is_Pauli() const;
    /**
     * \~japanese-en ゲートがクリフォードゲートかどうかを判定する
     * 
     * @return true クリフォードゲートである
     * @return false クリフォードゲートではない
     */
    bool is_Clifford() const;
    /**
     * \~japanese-en ゲートがFermionic Gaussianかどうかを判定する
     * 
     * @return true Fermionic Gaussianである
     * @return false Fermionic Gaussianではない
     */
    bool is_Gaussian() const;
    /**
     * \~japanese-en ゲートがparametricかどうかを判定する
     * 
     * @return true parametricである
     * @return false parametricではない
     */
    bool is_parametric() const;
    /**
     * \~japanese-en ゲート行列が対角行列かどうかを判定する
     * 
     * @return true 対角行列である
     * @return false 対角行列ではない
     */
    bool is_diagonal() const;

    /**
     * \~japanese-en ゲートのプロパティ値を取得する
     * 
     * ゲートのプロパティ値はゲートがパウリかどうかなどのゲート全体の性質の情報を持つ
     * @return プロパティ値
     */
    UINT get_property_value() const;

    /**
     * \~japanese-en ゲートがある添え字の量子ビットにおいて、与えられたパウリ演算子と可換かどうかを判定する。
     * 
     * @param qubit_index 量子ビットの添え字
     * @param pauli_type 比較するパウリ演算子。(I,X,Y,Z)が(0,1,2,3)に対応する。
     * @return true 可換である
     * @return false 可換ではない
     */
    bool commute_Pauli_at(UINT qubit_index, UINT pauli_type) const;

    /**
     * \~japanese-en 量子ゲートの名前を出力する。
     * 
     * @return ゲート名
     */
    virtual std::string get_name () const;

    /**
     * \~japanese-en 量子ゲートのデバッグ情報の文字列を生成する
     *
     * @return 生成した文字列
     */
    virtual std::string to_string() const;

    /**
     * \~japanese-en 量子回路のデバッグ情報を出力する。
     * 
     * @return 受け取ったストリーム
     */
    friend DllExport std::ostream& operator<<(std::ostream& os, const QuantumGateBase&);
    /**
     * \~japanese-en 量子回路のデバッグ情報を出力する。
     * 
     * @return 受け取ったストリーム
     */
    friend DllExport std::ostream& operator<<(std::ostream& os, const QuantumGateBase* gate);
};

