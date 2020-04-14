#pragma once

#include "gate.hpp"
#include "type.hpp"

/**
 * \~japanese-en 行列要素で自身が作用する内容を保持するクラス
 */
/**
 * \~english A class that holds the contents it will operate as matrix elements
 */
class DllExport QuantumGateSparseMatrix : public QuantumGateBase {
private:
	SparseComplexMatrix _matrix_element;     /**< list of elements of unitary matrix as 1D array with length dim*dim (only for dense gate))*/
public:
	/**
	 * \~japanese-en コンストラクタ
	 *
	 * 行列要素はコピーされるため、matrixは再利用できるが低速である
	 * @param target_qubit_index_list ターゲットとなる量子ビットの添え字のリスト
	 * @param matrix_element 行列要素
	 * @param control_qubit_index_list コントロールとなる量子ビットのリスト <code>control_value</code>はすべて1になる。
	 */
	/**
     	 * \~english Constructor
     	 * 
     	 * Matrix can be reused but the process is slow because matrix elements are copied
     	 * @param target_qubit_index_list List of target qubit subscripts
     	 * @param matrix_element Matrix element
    	 * @param target_qubit_index_list List of control qubit subscripts
     	 */
	QuantumGateSparseMatrix(const std::vector<UINT>& target_qubit_index_list, const SparseComplexMatrix& matrix_element, const std::vector<UINT>& control_qubit_index_list = {});

	/**
	 * \~japanese-en コンストラクタ
	 *
	 * 行列要素はswapされるため、matrixは再利用できないが高速である。
	 * @param target_qubit_index_list ターゲットとなる量子ビットの添え字のリスト
	 * @param matrix_element 行列要素
	 * @param control_qubit_index_list コントロールとなる量子ビットのリスト <code>control_value</code>はすべて1になる。
	 */
	/**
     	 * \~english Constructor
     	 * 
     	 * Matrix can be reused but the process is slow because matrix elements are copied
     	 * @param target_qubit_index_list List of target qubit subscripts
     	 * @param matrix_element Matrix element
    	 * @param target_qubit_index_list List of control qubit subscripts
     	 */
	QuantumGateSparseMatrix(const std::vector<UINT>& target_qubit_index_list, SparseComplexMatrix* matrix_element, const std::vector<UINT>& control_qubit_index_list = {});

	/**
	 * \~japanese-en コンストラクタ
	 *
	 * 行列要素はコピーされるため、matrixは再利用できるが低速である
	 * @param target_qubit_index_list ターゲットとなる量子ビットの情報のリスト
	 * @param matrix_element 行列要素
	 * @param control_qubit_index_list コントロールとなる量子ビットの情報のリスト
	 */
	/**
    	 * \~english Constructor
    	 * 
    	 * Matrix can be reused but the process is slow because matrix elements are copied
    	 * @param target_qubit_index_list List of target qubit subscripts
    	 * @param matrix_element Matrix element
    	 * @param target_qubit_index_list List of control qubit subscripts
    	 */
	QuantumGateSparseMatrix(const std::vector<TargetQubitInfo>& target_qubit_index_list, const SparseComplexMatrix& matrix_element, const std::vector<ControlQubitInfo>& control_qubit_index_list = {});

	/**
	 * \~japanese-en コンストラクタ
	 *
	 * 行列要素はswapされるため、matrixは再利用できないが高速である。
	 * @param target_qubit_index_list ターゲットとなる量子ビットの情報のリスト
	 * @param matrix_element 行列要素
	 * @param control_qubit_index_list コントロールとなる量子ビットの情報のリスト
	 */
	/**
    	 * \~english Constructor
    	 * 
    	 * Matrix can be reused but the process is slow because matrix elements are copied
    	 * @param target_qubit_index_list List of target qubit subscripts
    	 * @param matrix_element Matrix element
    	 * @param target_qubit_index_list List of control qubit subscripts
    	 */
	QuantumGateSparseMatrix(const std::vector<TargetQubitInfo>& target_qubit_index_list, SparseComplexMatrix* matrix_element, const std::vector<ControlQubitInfo>& control_qubit_index_list = {});

	/**
	 * \~japanese-en デストラクタ
	 */
	/**
	 * \~english Destructor
	 */
	virtual ~QuantumGateSparseMatrix() {};

	/**
	 * \~japanese-en コントロールの量子ビットを追加する
	 *
	 * <code>qubit_index</code>はゲートのターゲットやコントロールの値に含まれてはいけない。
	 * @param[in] qubit_index コントロールの量子ビットの添え字
	 * @param[in] control_value 基底の<code>qubit_index</code>が<code>control_value</code>である場合にのみゲートが作用する。
	 */
	/**
    	 * \~english Add control qubit
    	 * 
    	 * <code>qubit_index</code> must not be included in the gate target or control value.
    	 * @param[in] qubit_index Subscript of control qubit
    	 * @param[in] control_value The gate works only if the base <code>qubit_index</code> is <code>control_value</code>.
    	 */
	virtual void add_control_qubit(UINT qubit_index, UINT control_value);

	/**
	 * \~japanese-en ゲート行列にスカラー値をかける
	 *
	 * @param[in] value かける値
	 */
	/**
    	 * \~english Multiply gate matrix by scalar value
    	 *
    	 * @param[in] value Multiply value
    	 */
	virtual void multiply_scalar(CPPCTYPE value) {
		_matrix_element *= value;
	}

	/**
	 * \~japanese-en ゲートのプロパティを設定する
	 *
	 * @param[in] gate_property_ ゲートのプロパティ値
	 */
	/**
    	 * \~english Set gate properties
    	 * 
    	 * @param[in] gate_property_ Gate property values
    	 */
	virtual void set_gate_property(UINT gate_property_) {
		_gate_property = gate_property_;
	}

	/**
	 * \~japanese-en 量子状態に作用する
	 *
	 * @param[in,out] state 更新する量子状態
	 */
	/**
    	 * \~english Operate on quantum states
    	 * 
    	 * @param[in,out] state Update quantum state
    	 */
	virtual void update_quantum_state(QuantumStateBase* state) override;

	/**
	 * \~japanese-en 自身のコピーを作成する
	 *
	 * @return コピーされたゲートのインスタンス
	 */
	/**
    	 * \~english Make a copy of itself
    	 * 
    	 * @return Instance of the copied gate
    	 */
	virtual QuantumGateBase* copy() const override {
		return new QuantumGateSparseMatrix(*this);
	};

	/**
	 * \~japanese-en 自身の行列要素をセットする
	 *
	 * @param[out] matrix 行列要素をセットする行列の参照
	 */
	/**
    	 * \~english Set matrix elements of itself
    	 * 
    	 * @param[out] matrix Matrix reference to set the matrix elements
    	 */
	virtual void set_matrix(ComplexMatrix& matrix) const override {
		matrix = this->_matrix_element.toDense();
	}

	/**
	 * \~japanese-en 量子回路のデバッグ情報の文字列を生成する
	 *
	 * @return 生成した文字列
	 */
	/**
    	 * \~english Generate a string of debug information for a quantum circuit
    	 *
    	 * @return The generated string
    	 */
	virtual std::string to_string() const override;

	/**
	 * \~japanese-en ゲートの情報を文字列で出力する
	 *
	 * @param os 出力するストリーム
	 * @param gate 情報の出力を行うゲート
	 * @return 受け取ったストリーム
	 */
	/**
    	 * \~english Output gate information as a character string
    	 * 
    	 * @param os The stream to output
    	 * @param gate Gate that outputs information
    	 * @return Stream received
    	 */
	friend DllExport std::ostream& operator<<(std::ostream& os, const QuantumGateSparseMatrix& gate);

	/**
	 * \~japanese-en ゲートの情報を文字列で出力する
	 *
	 * @param os 出力するストリーム
	 * @param gate 情報の出力を行うゲート
	 * @return 受け取ったストリーム
	 */
	/**
    	 * \~english Output gate information as a character string
    	 * 
    	 * @param os The stream to output
    	 * @param gate Gate that outputs information
    	 * @return Stream received
    	 */
	friend DllExport std::ostream& operator<<(std::ostream& os, QuantumGateSparseMatrix* gate);

};
