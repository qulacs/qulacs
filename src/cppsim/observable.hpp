/**
 * @file Observable.hpp
 * @brief Definition and basic functions for Observable
 */


#pragma once

#include "type.hpp"
#include <vector>
#include <utility>
#include <string>

class QuantumStateBase;
class PauliOperator;

/**
 * \~japanese-en
 * @struct Observable
 * オブザーバブルの情報を保持するクラス。
 * PauliOperatorをリストとして持ち, 種々の操作を行う。解放時には保持しているPauliOperatorを全て解放する。
 */
class DllExport Observable {
private:
    //! list of multi pauli term
    std::vector<PauliOperator*> _operator_list;
    //! the number of qubits
    UINT _qubit_count;
public:
    /**
     * \~japanese-en
     * コンストラクタ。
     *
     * 空のオブザーバブルを作成する。
     * @param[in] qubit_count qubit数
     * @return Observableのインスタンス
     */
    Observable(UINT qubit_count);

    /**
     * \~japanese-en
     * デストラクタ。このとき、オブザーバブルが保持しているPauliOperatorは解放される。
     */
    virtual ~Observable();

    /**
     * \~japanese-en
     * PauliOperatorを内部で保持するリストの末尾に追加する。
     *
     * @param[in] mpt 追加するPauliOperatorのインスタンス
     */
    void add_operator(const PauliOperator* mpt);

    /**
     * \~japanese-en
     * パウリ演算子の文字列と係数の組をオブザーバブルに追加する。
     *
     * @param[in] coef pauli_stringで作られるPauliOperatorの係数
     * @param[in] pauli_string パウリ演算子と掛かるindexの組からなる文字列。(example: "X 1 Y 2 Z 5")
     */
    void add_operator(double coef, std::string pauli_string);

    /**
     * \~japanese-en
     * オブザーバブルが掛かるqubit数を返す。
     * @return オブザーバブルのqubit数
     */
    UINT get_qubit_count() const { return _qubit_count; }

    /**
     * \~japanese-en
     * オブザーバブルの行列表現の次元を返す。
     * @return オブザーバブルの次元
     */
    ITYPE get_state_dim() const { return (1ULL) << _qubit_count; }

    /**
     * \~japanese-en
     * オブザーバブルが保持するPauliOperatorの数を返す
     * @return オブザーバブルが保持するPauliOperatorの数
     */
    UINT get_term_count() const { return (UINT)_operator_list.size(); }

    /**
     * \~japanese-en
     * オブザーバブルの指定した添字に対応するPauliOperatorを返す
     * @param[in] index オブザーバブルが保持するPauliOperatorのリストの添字
     * @return 指定したindexにあるPauliOperator
     */
    const PauliOperator* get_term(UINT index) const { return _operator_list[index]; }

    /**
     * \~japanese-en
     * オブザーバブルが保持するPauliOperatorのリストを返す
     * @return オブザーバブルが持つPauliOperatorのリスト
     */
    std::vector<PauliOperator*> get_terms() const { return _operator_list;}

    /**
     * \~japanese-en
     * オブザーバブルのある量子状態に対応するエネルギー(期待値)を計算して返す
     *
     * @param[in] state 期待値をとるときの量子状態
     * @return 入力で与えた量子状態に対応するオブザーバブルの期待値
     */
    double get_expectation_value(const QuantumStateBase* state) const ;

    /**
     * \~japanese-en
     * オブザーバブルによってある状態が別の状態に移る遷移振幅を計算して返す
     *
     * @param[in] state_bra 遷移先の量子状態
     * @param[in] state_ket 遷移前の量子状態
     * @return 入力で与えた量子状態に対応するオブザーバブルの遷移振幅
     */
    CPPCTYPE get_transition_amplitude(const QuantumStateBase* state_bra, const QuantumStateBase* state_ket) const;

};

namespace observable{
    /**
     * \~japanese-en
     *
     * OpenFermionから出力されたオブザーバブルのテキストファイルを読み込んでObservableを生成します。オブザーバブルのqubit数はファイル読み込み時に、オブザーバブルの構成に必要なqubit数となります。
     *
     * @param[in] filename OpenFermion形式のオブザーバブルのファイル名
     * @return Observableのインスタンス
     **/
    DllExport Observable* create_observable_from_openfermion_file(std::string file_path);

    /**
     * \~japanese-en
     *
     * OpenFermionの出力テキストを読み込んでObservableを生成します。オブザーバブルのqubit数はファイル読み込み時に、オブザーバブルの構成に必要なqubit数となります。
     *
     * @param[in] filename OpenFermion形式のテキスト
     * @return Observableのインスタンス
     **/
    DllExport Observable* create_observable_from_openfermion_text(std::string text);

    /**
     * \~japanese-en
     * OpenFermion形式のファイルを読んで、対角なObservableと非対角なObservableを返す。オブザーバブルのqubit数はファイル読み込み時に、オブザーバブルの構成に必要なqubit数となります。
     *
     * @param[in] filename OpenFermion形式のオブザーバブルのファイル名
     */
    DllExport std::pair<Observable*, Observable*> create_split_observable(std::string file_path);
}
