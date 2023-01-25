#pragma once
/**
 * @file Observable.hpp
 * @brief Definition and basic functions for HermitianQuantumOperator
 */

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "general_quantum_operator.hpp"
#include "pauli_operator.hpp"
#include "type.hpp"

class QuantumStateBase;
class PauliOperator;
class GeneralQuantumOperator;

/**
 * \~japanese-en
 * @struct HermitianQuantumOperator
 * オブザーバブルの情報を保持するクラス。
 * PauliOperatorをリストとして持ち,
 * 種々の操作を行う。解放時には保持しているPauliOperatorを全て解放する。
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
    using GeneralQuantumOperator::GeneralQuantumOperator;

    /**
     * \~japanese-en
     * PauliOperatorを内部で保持するリストの末尾に追加する。
     *
     * @param[in] mpt 追加するPauliOperatorのインスタンス
     */
    void add_operator(const PauliOperator* mpt) override;

    /**
     * \~japanese-en
     * PauliOperatorを内部で保持するリストの末尾に追加する。
     *
     * @param[in] mpt 追加するPauliOperatorのインスタンス
     */
    void add_operator_move(PauliOperator* mpt) override;

    /**
     * \~japanese-en
     * PauliOperatorを内部で保持するリストの末尾に追加する。
     *
     * @param[in] mpt 追加するPauliOperatorのインスタンス
     */
    void add_operator_copy(const PauliOperator* mpt) override;

    /**
     * \~japanese-en
     * パウリ演算子の文字列と係数の組をオブザーバブルに追加する。
     *
     * @param[in] coef pauli_stringで作られるPauliOperatorの係数
     * @param[in] pauli_string
     * パウリ演算子と掛かるindexの組からなる文字列。(example: "X 1 Y 2 Z 5")
     */
    void add_operator(CPPCTYPE coef, std::string pauli_string) override;

    /**
     * \~japanese-en
     * HermitianQuantumOperatorのある量子状態に対応するエネルギー(期待値)を計算して返す
     *
     * @param[in] state 期待値をとるときの量子状態
     * @return 入力で与えた量子状態に対応するHermitianQuantumOperatorの期待値
     */
    CPPCTYPE get_expectation_value(
        const QuantumStateBase* state) const override;

    /**
     * \~japanese-en
     * GeneralQuantumOperator の基底状態の固有値を lanczos method により求める
     * (A - \mu I) の絶対値最大固有値を求めることで基底状態の固有値を求める．
     * @param[in] init_state 固有値を求めるための量子状態
     * @param[in] iter_count 計算の繰り返し回数
     * @param [in] mu 固有値をシフトするための係数
     * @return GeneralQuantumOperator の基底状態の固有値
     */
    CPPCTYPE solve_ground_state_eigenvalue_by_lanczos_method(
        QuantumStateBase* init_state, const UINT iter_count,
        const CPPCTYPE mu = 0.0) const;

    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual HermitianQuantumOperator* copy() const override {
        auto hermitian_quantum_operator =
            new HermitianQuantumOperator(this->get_qubit_count());
        for (auto pauli : this->get_terms()) {
            hermitian_quantum_operator->add_operator_copy(pauli);
        }
        return hermitian_quantum_operator;
    };

    /**
     * \~japanese-en
     * 文字列に変換する。
     */
    std::string to_string() const override;
};

namespace observable {
/**
 * \~japanese-en
 *
 * OpenFermionから出力されたオブザーバブルのテキストファイルを読み込んでHermitianQuantumOperatorを生成します。オブザーバブルのqubit数はファイル読み込み時に、オブザーバブルの構成に必要なqubit数となります。
 *
 * @param[in] file_path OpenFermion形式のオブザーバブルのファイルパス
 * @return Observableのインスタンス
 **/
DllExport HermitianQuantumOperator* create_observable_from_openfermion_file(
    std::string file_path);

/**
 * \~japanese-en
 *
 * OpenFermionの出力テキストを読み込んでObservableを生成します。オブザーバブルのqubit数はファイル読み込み時に、オブザーバブルの構成に必要なqubit数となります。
 *
 * @param[in] text OpenFermion形式のテキスト
 * @return Observableのインスタンス
 **/
DllExport HermitianQuantumOperator* create_observable_from_openfermion_text(
    const std::string& text);

/**
 * \~japanese-en
 * OpenFermion形式のファイルを読んで、対角なObservableと非対角なObservableを返す。オブザーバブルのqubit数はファイル読み込み時に、オブザーバブルの構成に必要なqubit数となります。
 *
 * @param[in] file_path OpenFermion形式のオブザーバブルのファイルパス
 */
DllExport std::pair<HermitianQuantumOperator*, HermitianQuantumOperator*>
create_split_observable(std::string file_path);

/**
 * \~japanese-en
 * ptreeからObservableを構築する
 *
 * @param[in] ptree ptree
 * @return Observableのインスタンス(ポインタ)
 */
DllExport HermitianQuantumOperator* from_ptree(
    const boost::property_tree::ptree& pt);
}  // namespace observable

using Observable = HermitianQuantumOperator;

/**
 * \~japanese-en
 * observable を対応する行列に変換する
 * @param [in] observable 行列に変換する observable
 * @return observable に対応する行列
 */
ComplexMatrix DllExport convert_observable_to_matrix(
    const HermitianQuantumOperator& observable);
