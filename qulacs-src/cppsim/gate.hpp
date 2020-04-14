
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
/**
 * \~english Basis class of quantum gate
 */
class DllExport QuantumGateBase {
protected:
    std::vector<TargetQubitInfo> _target_qubit_list; 
    std::vector<ControlQubitInfo> _control_qubit_list;
    UINT _gate_property=0;                            /**< \~japanese-en property of whole gate (e.g. Clifford or Pauli)*/
	/**< \~english property of whole gate (e.g. Clifford or Pauli)*/
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
    /**
     * \~english destructor
     */
    virtual ~QuantumGateBase() {};

	const std::vector<TargetQubitInfo>& target_qubit_list; /**< \~japanese-en ターゲット量子ビットのリスト */
	/**< \~english List of target qubit */
	const std::vector<ControlQubitInfo>& control_qubit_list; /**< \~japanese-en コントロール量子ビットのリスト */
	/**< \~english List of control qubit */

    /**
     * \~japanese-en ターゲット量子ビットの添え字のリストを取得する
     * 
     * @return 量子ビットの添え字のリスト
     */
    /**
     * \~english Obtain the list of target qubit subscripts
     * 
     * @return List of target qubit subscripts
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
    /**
     * \~english Obtain the list of control qubit subscripts
     * 
     * @return List of target qubit subscripts
     */
    std::vector<UINT> get_control_index_list() const {
        std::vector<UINT> res(control_qubit_list.size());
        for (UINT i = 0; i < control_qubit_list.size(); ++i) res[i] = control_qubit_list[i].index();
        return res;
    }

    /**
     * \~japanese-en 量子状態を更新する
     * 
     * @param state 更新する量子状態
     */
    /**
     * \~english Update quantum state
     * 
     * @param state Quantum state to be updated
     */
    virtual void update_quantum_state(QuantumStateBase* state)  = 0 ;
    /**
     * \~japanese-en 自身のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    /**
     * \~english Generate a deep copy of itself
     * 
     * @return Deep copy of itself
     */
    virtual QuantumGateBase* copy() const = 0;
    /**
     * \~japanese-en 自身のゲート行列をセットする
     * 
     * @param matrix 行列をセットする変数の参照
     */
    /**
     * \~english Set gate matrix of itself
     * 
     * @param matrix Reference variables to set matrix
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
    /**
     * \~english Determines whether the given gate <code> gate </code> is interchangeable with itself.
     * 
     * Even if it is determined to be non-commutative, it may actually be commutative.
     * If it is determined that commutation is possible, it is always commutative.
     * @param gate Gate to compare or commute
     * @return true Commutative
     * @return false Non-commutative
     */
    bool is_commute(const QuantumGateBase* gate) const;
    /**
     * \~japanese-en ゲートがパウリゲートかどうかを判定する
     * 
     * @return true パウリゲートである
     * @return false パウリゲートではない
     */
    /**
     * \~english Determine if a gate is Pauligate
     * 
     * @return true Pauligate
     * @return false Not Pauligate
     */
    bool is_Pauli() const;
    /**
     * \~japanese-en ゲートがクリフォードゲートかどうかを判定する
     * 
     * @return true クリフォードゲートである
     * @return false クリフォードゲートではない
     */
    /**
     * \~english Determine if a gate is Clifford gate
     * 
     * @return true Clifford gate
     * @return false Not Clifford gate
     */
    bool is_Clifford() const;
    /**
     * \~japanese-en ゲートがFermionic Gaussianかどうかを判定する
     * 
     * @return true Fermionic Gaussianである
     * @return false Fermionic Gaussianではない
     */
    /**
     * \~english Determine if a gate is Fermionic Gaussian
     * 
     * @return true Fermionic Gaussian
     * @return false Not Fermionic Gaussian
     */
    bool is_Gaussian() const;
    /**
     * \~japanese-en ゲートがparametricかどうかを判定する
     * 
     * @return true parametricである
     * @return false parametricではない
     */
    /**
     * \~english Determine if a gate is parametric
     * 
     * @return true Parametric
     * @return false Notparametric
     */
    bool is_parametric() const;
    /**
     * \~japanese-en ゲート行列が対角行列かどうかを判定する
     * 
     * @return true 対角行列である
     * @return false 対角行列ではない
     */
    /**
     * \~english Determine if a gate matrix is diagonal 
     * 
     * @return true Diagonal matrix
     * @return false Non-diagonal matrix
     */
    bool is_diagonal() const;

    /**
     * \~japanese-en ゲートのプロパティ値を取得する
     * 
     * ゲートのプロパティ値はゲートがパウリかどうかなどのゲート全体の性質の情報を持つ
     * @return プロパティ値
     */
    /**
     * \~english Get the property value of a gate
     * 
     * The gate property value contains information about the properties of entire gate, such as whether the gate is Pauli.
     * @return Property value
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
    /**
     * \~english Determine whether a gate is commutative with a given Pauli operator at a certain subscript of qubit.
     * 
     * @param qubit_index Subscript of qubit
     * @param pauli_type Pauli operator to compare. (I, X, Y, Z) corresponds to (0,1,2,3).
     * @return true Commutative
     * @return false Non-commutative
     */
    bool commute_Pauli_at(UINT qubit_index, UINT pauli_type) const;

    /**
     * \~japanese-en 量子ゲートの名前を出力する。
     * 
     * @return ゲート名
     */
    /**
     * \~english Output gate name
     * 
     * @return Gate name
     */
    virtual std::string get_name () const;

    /**
     * \~japanese-en 量子ゲートのデバッグ情報の文字列を生成する
     *
     * @return 生成した文字列
     */
    /**
     * \~english Generate a string of debug information of quantum gate
     *
     * @return Generated string
     */
    virtual std::string to_string() const;

    /**
     * \~japanese-en 量子回路のデバッグ情報を出力する。
     * 
     * @return 受け取ったストリーム
     */
    /**
     * \~english Outupt debug information of quantum gate
     * 
     * @return Received string
     */
    friend DllExport std::ostream& operator<<(std::ostream& os, const QuantumGateBase&);
    /**
     * \~japanese-en 量子回路のデバッグ情報を出力する。
     * 
     * @return 受け取ったストリーム
     */
    /**
     * \~english Outupt debug information of quantum gate
     * 
     * @return Received string
     */
    friend DllExport std::ostream& operator<<(std::ostream& os, const QuantumGateBase* gate);
};

