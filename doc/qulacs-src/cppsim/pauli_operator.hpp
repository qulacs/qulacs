
/**
 * @file pauli_operator.hpp
 * @brief Definition and basic functions for MultiPauliTerm
 */

#pragma once

#include "type.hpp"
#include <vector>
#include <cassert>
#include <iostream>

class QuantumStateBase;

/**
 * \~japanese-en
 * @struct SiglePauliOperator
 * 単一qubitに作用するパウリ演算子の情報を保持するクラス。
 * 作用するqubitの添字と自身のパウリ演算子の情報をもつ。
 */
/**
 * \~english
 * @struct SiglePauliOperator
 * A class that holds information on Pauli operators operating on single qubit.
 * It has information on the subscript of the qubit that operates on and its own Pauli operator.
 */
class DllExport SinglePauliOperator {
protected:
    UINT _index;
    UINT _pauli_id;
public:
    /**
     * \~japanese-en
     * コンストラクタ
     *
     * 添字とパウリ演算子からインスタンスを生成する。
     *
     * @param[in] index_ 作用するqubitの添字
     * @param[in] pauli_id_ パウリ演算子を表す整数。(I,X,Y,Z)が(0,1,2,3)に対応する。
     * @return 新しいインスタンス
     */
    /**
     * \~english
     * Construct
     *
     * Creating an instance from a subscript and Pauli operator.
     *
     * @param[in] index_ Subscript of the qubit to operate on
     * @param[in] pauli_id_ An integer representing the Pauli operator. (I,X,Y,Z) corresponds to (0,1,2,3).
     * @return New instance
     */
    SinglePauliOperator(UINT index_, UINT pauli_id_) : _index(index_), _pauli_id(pauli_id_) {
		if (pauli_id_ > 3) {
			std::cerr << "Error: SinglePauliOperator(UINT, UINT): index must be either of 0,1,2,3" << std::endl;
		}
    };

    /**
     * \~japanese-en
     * 自身が作用する添字を返す
     *
     * @return 自身が作用する添字
     */
    /**
     * \~english
     * Return the index on which it operates
     *
     * @return The index on which it operates
     */
    UINT index() const { return _index; }

    /**
     * \~japanese-en
     * 自身を表すパウリ演算子を返す
     *
     * @return 自身のもつパウリ演算子を表す整数。(I,X,Y,Z)が(0,1,2,3)に対応する。
     */
    /**
     * \~english
     * Return the Pauli operator representing itself
     *
     * @return An integer representing the Pauli operator. (I,X,Y,Z) corresponds to (0,1,2,3).
     */
    UINT pauli_id() const { return _pauli_id; }
};

/**
 * \~japanese-en
 * @struct PauliOperator
 * 複数qubitに作用するパウリ演算子の情報を保持するクラス。
 * SinglePauliOperatorをリストとして持ち, 種々の操作を行う。
 */
/**
 * \~english
 * @struct PauliOperator
 * A class that retains information on Pauli operators operating on multiple qubits.
 * Hold SinglePauliOperator as a list and perform various operations.
 */
class DllExport PauliOperator{
private:
    std::vector<SinglePauliOperator> _pauli_list;
    CPPCTYPE _coef;
public:
    /**
     * \~japanese-en
     * 自身の保持するパウリ演算子が作用する添字のリストを返す
     *
     * それぞれの添字に作用する演算子は PauliOperator::get_pauli_id_listで得られる添字のリストの対応する場所から得られる。
     * 
     * @return 自身の保持するパウリ演算子が作用する添字のリスト。
     */
    /**
     * \~english
     * Return a list of indices on which the Pauli operator held operates
     *
     * The operator that operates on each subscript is obtained from the corresponding place in the list of subscripts obtained by PauliOperator::get_pauli_id_list.
     * 
     * @return List of indices on which the Pauli operator held operates.
     */
    std::vector<UINT> get_index_list() const {
        std::vector<UINT> res;
        for (auto val : _pauli_list) res.push_back(val.index());
        return res;
    }

    /**
     * \~japanese-en
     * 自身が保持するパウリ演算子を返す。
     *
     * それぞれが作用するqubitは PauliOperator::get_index_listで得られる添字のリストの対応する場所から得られる。
     *
     * @return 自身の保持するパウリ演算子のリスト。(I,X,Y,Z)が(0,1,2,3)に対応する。
     */
    /**
     * \~english
     * Return the Pauli operator held
     *
     * The operator that operates on each subscript is obtained from the corresponding place in the list of subscripts obtained by PauliOperator::get_index_list.
     * 
     * @return List of the Pauli operator held. (I,X,Y,Z) corresponds to (0,1,2,3).
     */
    std::vector<UINT> get_pauli_id_list() const {
        std::vector<UINT> res;
        for (auto val : _pauli_list) res.push_back(val.pauli_id());
        return res;
    }

    /**
     * \~japanese-en
     * コンストラクタ
     *
     * 係数をとって空のインスタンスを返す。
     *
     * @param[in] coef 係数。デフォルトは1.0
     * @return 係数がcoefの空のインスタンス
     */
    /**
     * \~english
     * Constructor
     *
     * Take the coefficient and return an empty instance
     *
     * @param[in] coef Coefficient. Default value is 1.0
     * @return An empty instance with coefficient coef
     */
    PauliOperator(CPPCTYPE coef=1.): _coef(coef){};

    /**
     * \~japanese-en
     * コンストラクタ
     *
     * パウリ演算子とその添字からなる文字列と、その係数から複数qubitに掛かるパウリ演算子を作成する
     *
     * @param[in] strings Pauli演算子とその掛かるindex. "X 1 Y 2 Z 5"のようにスペース区切りの文字列
     * @param[in] coef 演算子の係数
     * @return 入力のパウリ演算子と係数をもつPauliOpetatorのインスタンス
     */
    /**
     * \~english
     * Constructor
     * Creating a Pauli operator over multiple qubits from a string consisting of Pauli operator and its subscript along with the coefficients
     *
     * @param[in] strings Pauli operator and its operating index, as a space separated string, such as "X 1 Y 2 Z 5".
     * @param[in] coef Coefficient of operator 
     * @return Instance of PauliOperator with the input Pauli operator and its coefficient
     */
    PauliOperator(std::string strings, CPPCTYPE coef=1.);

    /**
     * \~japanese-en
     * コンストラクタ
     *
     * パウリ演算子の文字列と添字のリスト、係数からPauliOperatorのインスタンスを生成する。
     * このとき入力として与える演算子と添字のリストは、i番目の演算子にi番目の添字が対応する。
     *
     * @param[in] target_qubit_index_list Pauli_operator_type_listで与えるパウリ演算子が掛かるqubitを指定する添字のリスト。
     * @param[in] Pauli_operator_type_list パウリ演算子の文字列。(example: "XXYZ")
     * @param[in] coef 係数
     * @return 入力として与えたパウリ演算子のリストと添字のリスト、係数から生成されるPauliOperatorのインスタンス
     */
     /**
     * \~japanese-en
     * Constructor
     *
     * Create PauliOperator instance from Pauli operator string, index list and coefficient.
     * In the list of operators and subscripts given as inputs, the i-th index corresponding to the i-th operator
     *
     * @param[in] target_qubit_index_list A list of indices specifying the qubit on which the Pauli operator given in Pauli_operator_type_list is to be applied on
     * @param[in] Pauli_operator_type_list String of Pauli operator. (example: "XXYZ")
     * @param[in] coef Coefficient
     * @return Instance of PauliOperator generated from the list of Pauli operator and list of indices and coefficients given as input
     */
    PauliOperator(const std::vector<UINT>& target_qubit_index_list, std::string Pauli_operator_type_list, CPPCTYPE coef = 1.);

    /**
     * \~japanese-en
     * コンストラクタ
     *
     * 配列の添字に作用するパウリ演算子と係数からインスタンスを生成する。
     * @param[in] pauli_list 配列の添字に対応するqubitに作用するパウリ演算子のリスト
     * @param[in] coef 係数
     * @return pauli_listの添字に対応するqubitに作用するパウリ演算子と係数をもつインスタンス
     */
    /**
     * \~english
     * Constructor
     *
     * Creating an instance from the Pauli operator and coefficients operating on the array index.
     * @param[in] pauli_list List of Pauli operators operating on qubits corresponding to array index
     * @param[in] coef Coefficient
     * @return An instance with coefficients and Pauli operators operating on the qubit corresponding to the index of pauli_list
     */
    PauliOperator(const std::vector<UINT>& pauli_list, CPPCTYPE coef = 1.);

    /**
     * \~japanese-en
     * コンストラクタ
     *
     * パウリ演算子のリストと添字のリスト、係数からPauliOperatorのインスタンスを生成する。
     * このとき入力として与える演算子と添字のリストは、リストの同じ添字の場所にあるものが対応する。
     *
     * @param[in] target_qubit_index_list Pauli_operator_type_listで与えるパウリ演算子が掛かるqubitを指定する添字のリスト
     * @param[in] target_qubit_pauli_list パウリ演算子の符号なし整数リスト。(I,X,Y,Z)が(0,1,2,3)に対応する。
     * @param[in] coef 係数
     * @return 入力として与えたパウリ演算子のリストと添字のリスト、係数から生成されるPauliOperatorのインスタンス
     */
    /**
     * \~english
     * Constructor
     *
     * Creating an instance of PauliOperator from the list of Pauli operators, list of index and coefficients.
     * The list of operators and indeces given as inputs corresponds to that at the same index position in the list.
     *
     * @param[in] target_qubit_index_list A list of indices specifying the qubits on which the Pauli operator given by Pauli_operator_type_list operates.
     * @param[in] target_qubit_pauli_list Integer list of Pauli operator without names. (I,X,Y,Z) correspond to (0,1,2,3).
     * @param[in] coef Coefficient
     * @return Instance of PauliOperator generated from the list of Pauli operator and list of indices and coefficients given as input
     */
    PauliOperator(const std::vector<UINT>& target_qubit_index_list, const std::vector<UINT>& target_qubit_pauli_list, CPPCTYPE coef=1.);


    /**
     * \~japanese-en
     * 自身の係数を返す
     *
     * @return 自身の係数
     */
    /**
     * \~english
     * Return coefficients of itself
     *
     * @return Coefficients of itselt
     */
    virtual CPPCTYPE get_coef() const { return _coef; }

    virtual ~PauliOperator(){};

    /**
     * \~japanese-en
     * 指定した添字のqubitに作用するSinglePauliOperatorを自身が保持するリストの末尾に追加する。
     *
     * @param[in] qubit_index 作用するqubitの添字
     * @param[in] pauli_type パウリ演算子。(I,X,Y,Z)が(0,1,2,3)に対応する。
     */
    /**
     * \~english
     * Adding a SinglePauliOperator operating on the qubit with the specified index to the end of the list it holds.
     *
     * @param[in] qubit_index Index of qubit to operate
     * @param[in] pauli_type Pauli operator.(I,X,Y,Z) correspond to (0,1,2,3).
     */
    virtual void add_single_Pauli(UINT qubit_index, UINT pauli_type);

    /**
     * \~japanese-en
     * 量子状態に対応するパウリ演算子の期待値を計算する
     *
     * @param[in] state 期待値をとるときの量子状態
     * @return stateに対応する期待値
     */
    /**
     * \~english
     * Calculate and return the expected value of Pauli operator corresponding to a quantum state
     *
     * @param[in] state Quantum state when taking expected value
     * @return Expected value of corresponding quantum state
     */
    virtual CPPCTYPE get_expectation_value(const QuantumStateBase* state) const;

    /**
     * \~japanese-en
     * 量子状態に対応するパウリ演算子の遷移振幅を計算する
     *
     * @param[in] state_bra 遷移先の量子状態
     * @param[in] state_ket 遷移元の量子状態
     * @return state_bra, state_ketに対応する遷移振幅
     */
    /**
     * \~english
     * Calculate and return the transition amplitude of Pauli operator corresponding to a quantum state.
     *
     * @param[in] state_bra Quantum state after transition
     * @param[in] state_ket Quantum state before transition
     * @return Transition amplitude corresponding to state_bra, state_ket
     */
    virtual CPPCTYPE get_transition_amplitude(const QuantumStateBase* state_bra, const QuantumStateBase* state_ket) const;

    /**
     * \~japanese-en
     * 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    /**
     * \~english Generate a deep copy of itself
     * 
     * @return Deep copy of itself
     */
    virtual PauliOperator* copy() const;

};



