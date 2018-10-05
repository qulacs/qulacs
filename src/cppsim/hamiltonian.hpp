/**
 * @file Hamiltonian.hpp
 * @brief Definition and basic functions for Hamiltonian
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
 * @struct Hamiltonian
 * ハミルトニアンの情報を保持するクラス。
 * PauliOperatorをリストとして持ち, 種々の操作を行う。解放時には保持しているPauliOperatorを全て解放する。
 */
class DllExport Hamiltonian {
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
	 * 空のハミルトニアンを作成する。
	 * @param[in] qubit_count qubit数
	 * @return Hamiltonianのインスタンス
	 */
	Hamiltonian(UINT qubit_count);

	/**
	 * \~japanese-en
	 * コンストラクタ。
	 *
	 * OpenFermionから出力されたハミルトニアンのテキストファイルを読み込んでHamiltonianを生成します。
	 *
	 * @param[in] filename OpenFermion形式のハミルトニアンのファイル名
	 * @return Hamiltonianのインスタンス
	 **/
	Hamiltonian(std::string filename);

	/**
     * \~japanese-en
	 * デストラクタ。このとき、ハミルトニアンが保持しているPauliOperatorは解放される。
	 */
	virtual ~Hamiltonian();

	/**
     * \~japanese-en
	 * PauliOperatorを内部で保持するリストの末尾に追加する。
	 *
	 * @param[in] mpt 追加するPauliOperatorのインスタンス
	 */
	void add_operator(PauliOperator* mpt);

    /**
     * \~japanese-en
     * パウリ演算子の文字列と係数の組をハミルトニアンに追加する。
     *
     * @param[in] coef pauli_stringで作られるPauliOperatorの係数
     * @param[in] pauli_string パウリ演算子と掛かるindexの組からなる文字列。(example: "X 1 Y 2 Z 5")
     */
	void add_operator(double coef, std::string pauli_string);

	/**
     * \~japanese-en
     * ハミルトニアンが掛かるqubit数を返す。
	 * @return ハミルトニアンのqubit数
	 */
	UINT get_qubit_count() const { return _qubit_count; }

	/**
     * \~japanese-en
     * ハミルトニアンの行列表現の次元を返す。
	 * @return ハミルトニアンの次元
	 */
	ITYPE get_state_dim() const { return (1ULL) << _qubit_count; }

    /**
     * \~japanese-en
     * ハミルトニアンが保持するPauliOperatorの数を返す
	 * @return ハミルトニアンが保持するPauliOperatorの数
	 */
	UINT get_term_count() const { return (UINT)_operator_list.size(); }

    /**
     * \~japanese-en
	 * ハミルトニアンの指定した添字に対応するPauliOperatorを返す
     * @param[in] index ハミルトニアンが保持するPauliOperatorのリストの添字
	 * @return 指定したindexにあるPauliOperator
	 */
	const PauliOperator* get_term(UINT index) const { return _operator_list[index]; }

	/**
     * \~japanese-en
	 * ハミルトニアンが保持するPauliOperatorのリストを返す
	 * @return ハミルトニアンが持つPauliOperatorのリスト
	 */
	std::vector<PauliOperator*> get_terms() const { return _operator_list;}

    /**
     * \~japanese-en
     * ハミルトニアンのある量子状態に対応するエネルギー(期待値)を計算して返す
     *
     * @param[in] state 期待値をとるときの量子状態
     * @return 入力で与えた量子状態に対応するハミルトニアンの期待値
     */
	double get_expectation_value(const QuantumStateBase* state) const ;

	/**
     * \~japanese-en
     * OpenFermion形式のファイルを読んで、対角なHamiltonianと非対角なHamiltonianを返す。ハミルトニアンのqubit数はファイル読み込み時に、ハミルトニアンの構成に必要なqubit数となります。
     *
	 * @param[in] filename OpenFermion形式のハミルトニアンのファイル名
     */
	static std::pair<Hamiltonian*, Hamiltonian*> get_split_hamiltonian(std::string filename);
};
