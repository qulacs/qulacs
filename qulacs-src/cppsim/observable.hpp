/**
 * @file Observable.hpp
 * @brief Definition and basic functions for HermitianQuantumOperator
 */


#pragma once

#include "type.hpp"
#include <iostream>
#include <vector>
#include <utility>
#include <string>
#include "pauli_operator.hpp"
#include "general_quantum_operator.hpp"

class QuantumStateBase;
class PauliOperator;
class GeneralQuantumOperator;

/**
 * \~japanese-en
 * @struct HermitianQuantumOperator
 * オブザーバブルの情報を保持するクラス。
 * PauliOperatorをリストとして持ち, 種々の操作を行う。解放時には保持しているPauliOperatorを全て解放する。
 */
/**
 * \~english
 * @struct HermitianQuantumOperator
 * A class that retains observable information.
 * It has PauliOperator as a list and performs various operations. At the time of release, all the held PauliOperators are released.
 */
class DllExport HermitianQuantumOperator : public GeneralQuantumOperator {
public:
    /**
     * \~japanese-en
     * コンストラクタ。
     *
     * 空の HermitianQuantumOperator を作成する。
     * @param[in] qubit_count qubit数
     * @return HermitianQuantumOperatorのインスタンス
     */
    /**
     * \~english
     * Constructor
     *
     * Creating empty HermitianQuantumOperator
     * @param[in] qubit_count Number of qubit
     * @return Instance of HermitianQuantumOperator
     */
    using GeneralQuantumOperator::GeneralQuantumOperator;

    /**
     * \~japanese-en
     * PauliOperatorを内部で保持するリストの末尾に追加する。
     *
     * @param[in] mpt 追加するPauliOperatorのインスタンス
     */    
    /**
     * \~english
     * Adding PauliOperator to the end of the internally stored list.
     *
     * @param[in] mpt Instance of PauliOperator added
     */ 
    void add_operator(const PauliOperator* mpt);

    /**
     * \~japanese-en
     * パウリ演算子の文字列と係数の組をオブザーバブルに追加する。
     *
     * @param[in] coef pauli_stringで作られるPauliOperatorの係数
     * @param[in] pauli_string パウリ演算子と掛かるindexの組からなる文字列。(example: "X 1 Y 2 Z 5")
     */
    /**
     * \~english
     * Add a Pauli operator string and coefficient pair to Observable
     *
     * @param[in] coef Coefficient of PauliOperator created from pauli_string
     * @param[in] pauli_string Character string consisting of a pair of Pauli operator and index.(example: "X 1 Y 2 Z 5")
     */
    void add_operator(CPPCTYPE coef, std::string pauli_string);

    /**
     * \~japanese-en
     * HermitianQuantumOperatorのある量子状態に対応するエネルギー(期待値)を計算して返す
     *
     * @param[in] state 期待値をとるときの量子状態
     * @return 入力で与えた量子状態に対応するHermitianQuantumOperatorの期待値
     */
    /**
     * \~english
     * Compute and return the energy (expected value) corresponding to a quantum state of HermitianQuantumOperator
     *
     * @param[in] state Quantum state when taking expected value
     * @return Expected value of HermitianQuantumOperator corresponding to quantum state given as input
     */
    CPPCTYPE get_expectation_value(const QuantumStateBase* state) const ;
};

namespace observable{
    /**
     * \~japanese-en
     *
     * OpenFermionから出力されたオブザーバブルのテキストファイルを読み込んでHermitianQuantumOperatorを生成します。オブザーバブルのqubit数はファイル読み込み時に、オブザーバブルの構成に必要なqubit数となります。
     *
     * @param[in] filename OpenFermion形式のオブザーバブルのファイル名
     * @return Observableのインスタンス
     **/
    /**
     * \~english
     * Generates a HermitianQuantumOperator by reading the text file of Observable output from OpenFermion. The number of qubits of Observable is the number of qubits required for configuring Observable when reading the file.
     *
     * @param[in] filename File name of Observable in form of OpenFermion
     * @return Instance of Observable
     **/
    DllExport HermitianQuantumOperator* create_observable_from_openfermion_file(std::string file_path);

    /**
     * \~japanese-en
     *
     * OpenFermionの出力テキストを読み込んでObservableを生成します。オブザーバブルのqubit数はファイル読み込み時に、オブザーバブルの構成に必要なqubit数となります。
     *
     * @param[in] filename OpenFermion形式のテキスト
     * @return Observableのインスタンス
     **/
    /**
     * \~english
     * Generates an Observable by reading the text file output from OpenFermion. The number of qubits of Observable is the number of qubits required for configuring Observable when reading the file.
     *
     * @param[in] filename Text in form of OpenFermion
     * @return Instance of Observable
     **/
    DllExport HermitianQuantumOperator* create_observable_from_openfermion_text(std::string text);

    /**
     * \~japanese-en
     * OpenFermion形式のファイルを読んで、対角なObservableと非対角なObservableを返す。オブザーバブルのqubit数はファイル読み込み時に、オブザーバブルの構成に必要なqubit数となります。
     *
     * @param[in] filename OpenFermion形式のオブザーバブルのファイル名
     */
    /**
     * \~english
     * Reads a file in OpenFermion format and returns a diagonal Observable and a non-diagonal Observable. The number of qubits of Observable is the number of qubits required for configuring Observable when reading the file.
     *
     * @param[in] filename File name of Observable in form of OpenFermion
     */
    DllExport std::pair<HermitianQuantumOperator*, HermitianQuantumOperator*> create_split_observable(std::string file_path);

}

typedef HermitianQuantumOperator Observable;
