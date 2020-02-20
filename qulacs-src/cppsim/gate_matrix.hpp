#pragma once

#include "gate.hpp"
#include "type.hpp"

/**
 * \~japanese-en 行列要素で自身が作用する内容を保持するクラス
 */
class DllExport QuantumGateMatrix : public QuantumGateBase{
private:
    ComplexMatrix _matrix_element;     /**< list of elements of unitary matrix as 1D array with length dim*dim (only for dense gate))*/
public:
    /**
     * \~japanese-en コンストラクタ
     * 
     * 行列要素はコピーされるため、matrixは再利用できるが低速である
     * @param target_qubit_index_list ターゲットとなる量子ビットの添え字のリスト
     * @param matrix_element 行列要素
     * @param control_qubit_index_list コントロールとなる量子ビットのリスト <code>control_value</code>はすべて1になる。
     */
    QuantumGateMatrix(const std::vector<UINT>& target_qubit_index_list, const ComplexMatrix& matrix_element, const std::vector<UINT>& control_qubit_index_list = {});

    /**
     * \~japanese-en コンストラクタ
     * 
     * 行列要素はswapされるため、matrixは再利用できないが高速である。
     * @param target_qubit_index_list ターゲットとなる量子ビットの添え字のリスト
     * @param matrix_element 行列要素
     * @param control_qubit_index_list コントロールとなる量子ビットのリスト <code>control_value</code>はすべて1になる。
     */
    QuantumGateMatrix(const std::vector<UINT>& target_qubit_index_list, ComplexMatrix* matrix_element, const std::vector<UINT>& control_qubit_index_list = {});

    /**
     * \~japanese-en コンストラクタ
     * 
     * 行列要素はコピーされるため、matrixは再利用できるが低速である
     * @param target_qubit_index_list ターゲットとなる量子ビットの情報のリスト
     * @param matrix_element 行列要素
     * @param control_qubit_index_list コントロールとなる量子ビットの情報のリスト
     */
    QuantumGateMatrix(const std::vector<TargetQubitInfo>& target_qubit_index_list, const ComplexMatrix& matrix_element, const std::vector<ControlQubitInfo>& control_qubit_index_list = {});

    /**
     * \~japanese-en コンストラクタ
     * 
     * 行列要素はswapされるため、matrixは再利用できないが高速である。
     * @param target_qubit_index_list ターゲットとなる量子ビットの情報のリスト
     * @param matrix_element 行列要素
     * @param control_qubit_index_list コントロールとなる量子ビットの情報のリスト
     */
    QuantumGateMatrix(const std::vector<TargetQubitInfo>& target_qubit_index_list, ComplexMatrix* matrix_element, const std::vector<ControlQubitInfo>& control_qubit_index_list = {});

    /**
     * \~japanese-en デストラクタ
     */
    virtual ~QuantumGateMatrix() {};

    /**
     * \~japanese-en コントロールの量子ビットを追加する
     * 
     * <code>qubit_index</code>はゲートのターゲットやコントロールの値に含まれてはいけない。
     * @param[in] qubit_index コントロールの量子ビットの添え字
     * @param[in] control_value 基底の<code>qubit_index</code>が<code>control_value</code>である場合にのみゲートが作用する。
     */
    virtual void add_control_qubit(UINT qubit_index, UINT control_value);

    /**
     * \~japanese-en ゲート行列にスカラー値をかける
     *
     * @param[in] value かける値
     */
    virtual void multiply_scalar(CPPCTYPE value) {
        _matrix_element *= value;
    }

    /**
     * \~japanese-en ゲートのプロパティを設定する
     * 
     * @param[in] gate_property_ ゲートのプロパティ値
     */
    virtual void set_gate_property(UINT gate_property_) {
        _gate_property = gate_property_;
    }

    /**
     * \~japanese-en 量子状態に作用する
     * 
     * @param[in,out] state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override;

    /**
     * \~japanese-en 自身のコピーを作成する
     * 
     * @return コピーされたゲートのインスタンス
     */
    virtual QuantumGateBase* copy() const override{
        return new QuantumGateMatrix(*this);
    };

    /**
     * \~japanese-en 自身の行列要素をセットする
     * 
     * @param[out] matrix 行列要素をセットする行列の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override{
        matrix = this->_matrix_element;
    }

    /**
     * \~japanese-en 量子回路のデバッグ情報の文字列を生成する
     *
     * @return 生成した文字列
     */
    virtual std::string to_string() const override;

    /**
     * \~japanese-en ゲートの情報を文字列で出力する
     * 
     * @param os 出力するストリーム
     * @param gate 情報の出力を行うゲート
     * @return 受け取ったストリーム
     */
    friend DllExport std::ostream& operator<<(std::ostream& os, const QuantumGateMatrix& gate);

    /**
     * \~japanese-en ゲートの情報を文字列で出力する
     * 
     * @param os 出力するストリーム
     * @param gate 情報の出力を行うゲート
     * @return 受け取ったストリーム
     */
    friend DllExport std::ostream& operator<<(std::ostream& os, QuantumGateMatrix* gate);

};
