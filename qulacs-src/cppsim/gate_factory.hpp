/**
 * @file gate_factory.hpp
 *
 * \~japanese-en ゲートを生成するための関数
 * \~english Functions to generate gate
 */
#pragma once

#include "type.hpp"
#include "gate.hpp"
#include "gate_general.hpp"
#include <vector>
#include <string>

namespace gate{
    /**
     * \~japanese-en 量子ゲートを文字列から生成する。
     * 
     * ゲートを生成するための文字列は以下の通り
     * Identity     :   I \<index\>
     * X            :   X \<index\>
     * Y            :   Y \<index\>
     * Z            :   Z \<index\>
     * H            :   H \<index\>
     * S            :   S \<index\>
     * Sdag         :   Sdag \<index\>
     * T            :   T \<index\>
     * Tdag         :   Tdag \<index\>
     * CNOT,CX      :   CNOT \<control\> \<target\>, or CX \<control\> \<target\>
     * CZ           :   CZ \<control\> \<target\>
     * SWAP         :   SWAP \<target1\> \<target2\>
     * U1           :   U1 \<index\> \<angle1\>
     * U2           :   U2 \<index\> \<angle1\> \<angle2\>
     * U3           :   U3 \<index\> \<angle1\> \<angle2\> \<angle3\>
     * Rot X        :   RX \<index\> \<angle1\>
     * Rot Y        :   RY \<index\> \<angle1\>
     * Rot Z        :   RZ \<index\> \<angle1\>
     * DifRot X     :   RDX \<index\>
     * DifRot Y     :   RDY \<index\>
     * DifRot Z     :   RDZ \<index\>
     * MultiRot     :   RM \<paulistr\> \<index1\> \<index2\> ... \<theta\> (for example: "RM XYZ 2 3 1 0.123")
     * DifMultiRot  :   RDM \<paulistr\> \<index1\> \<index2\> ...  (for example: "RDM XYZ 2 3 1")
     * general U    :   U \<index_count\> \<index1\> \<index2\> ... \<element1_real\> \<element1_imag\> \<element2_real\> ...
     * @param[in] gate_string ゲートを生成する文字列
     * @return 作成されたゲートのインスタンス
     */
	
    /**
     * \~english Generate a quantum gate from a string.
     * 
     * The tring for generating the gate is as follows:
     * Identity     :   I \<index\>
     * X            :   X \<index\>
     * Y            :   Y \<index\>
     * Z            :   Z \<index\>
     * H            :   H \<index\>
     * S            :   S \<index\>
     * Sdag         :   Sdag \<index\>
     * T            :   T \<index\>
     * Tdag         :   Tdag \<index\>
     * CNOT,CX      :   CNOT \<control\> \<target\>, or CX \<control\> \<target\>
     * CZ           :   CZ \<control\> \<target\>
     * SWAP         :   SWAP \<target1\> \<target2\>
     * U1           :   U1 \<index\> \<angle1\>
     * U2           :   U2 \<index\> \<angle1\> \<angle2\>
     * U3           :   U3 \<index\> \<angle1\> \<angle2\> \<angle3\>
     * Rot X        :   RX \<index\> \<angle1\>
     * Rot Y        :   RY \<index\> \<angle1\>
     * Rot Z        :   RZ \<index\> \<angle1\>
     * DifRot X     :   RDX \<index\>
     * DifRot Y     :   RDY \<index\>
     * DifRot Z     :   RDZ \<index\>
     * MultiRot     :   RM \<paulistr\> \<index1\> \<index2\> ... \<theta\> (for example: "RM XYZ 2 3 1 0.123")
     * DifMultiRot  :   RDM \<paulistr\> \<index1\> \<index2\> ...  (for example: "RDM XYZ 2 3 1")
     * general U    :   U \<index_count\> \<index1\> \<index2\> ... \<element1_real\> \<element1_imag\> \<element2_real\> ...
     * @param[in] gate_string Strint to generate the gate
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* create_quantum_gate_from_string(std::string gate_string);

    /**
     * \~japanese-en Identityゲートを作成する。
     * 
     * 作用しても状態は変わらないが、ノイズなどが付与された際の挙動が異なる。
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate Identity gate
     * 
     * Identity gate does not change the state, but behaves differently when noise is added.
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* Identity(UINT qubit_index);

    /**
     * \~japanese-en \f$X\f$ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$X\f$ gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* X(UINT qubit_index);

    /**
     * \~japanese-en \f$Y\f$ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$Y\f$ gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* Y(UINT qubit_index);

    /**
     * \~japanese-en \f$Z\f$ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$Z\f$ gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* Z(UINT qubit_index);

    /**
     * \~japanese-en Hadamardゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate Hadamard gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* H(UINT qubit_index);

    /**
     * \~japanese-en \f$S\f$ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$S\f$ gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* S(UINT qubit_index);

    /**
     * \~japanese-en \f$S^{\dagger}\f$ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$S^{\dagger}\f$ gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* Sdag(UINT qubit_index);

    /**
     * \~japanese-en \f$T\f$ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$T\f$ gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* T(UINT qubit_index);

    /**
     * \~japanese-en \f$T^{\dagger}\f$ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$T^{\dagger}\f$ gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* Tdag(UINT qubit_index);

    /**
     * \~japanese-en \f$\sqrt{X}\f$ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$\sqrt{X}\f$ gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* sqrtX(UINT qubit_index);

    /**
     * \~japanese-en \f$\sqrt{X}^{\dagger}\f$ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$\sqrt{X}^{\dagger}\f$ gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* sqrtXdag(UINT qubit_index);

    /**
     * \~japanese-en \f$\sqrt{Y}\f$ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$\sqrt{Y}\f$ gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* sqrtY(UINT qubit_index);

    /**
     * \~japanese-en \f$\sqrt{Y}^{\dagger}\f$ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$\sqrt{Y}^{\dagger}\f$ gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* sqrtYdag(UINT qubit_index);

    /**
     * \~japanese-en <code>qubit_index</code>を0へ射影するゲートを作成する
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate a gate that projects <code> qubit_index </ code> to 0
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* P0(UINT qubit_index);

    /**
     * \~japanese-en <code>qubit_index</code>を1へ射影するゲートを作成する
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate a gate that projects <code> qubit_index </ code> to 1
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* P1(UINT qubit_index);

    /**
     * \~japanese-en OpenQASMのU1ゲートを作成する。
     * 
     * 具体的なゲートについてはOpenQASMのマニュアルを参照
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @param[in] lambda 回転角の第一引数
     * @return 作成されたゲートのインスタンス
     */

    /**
     * \~english Generate U1 gate of OpenQASM
     * 
     * See the OpenQASM manual for details of gates
     * @param[in] qubit_index Subscript of target qubit
     * @param[in] lambda The first argument of rotation angle
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* U1(UINT qubit_index, double lambda);

    /**
     * \~japanese-en OpenQASMのU2ゲートを作成する。
     * 
     * 具体的なゲートについてはOpenQASMのマニュアルを参照
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @param[in] lambda 回転角の第一引数
     * @param[in] phi 回転角の第二引数
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate U2 gate of OpenQASM
     * 
     * See the OpenQASM manual for details of gates
     * @param[in] qubit_index Subscript of target qubit
     * @param[in] lambda The first argument of rotation angle
     * @param[in] phi The second argument of rotation angle
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* U2(UINT qubit_index, double phi, double lambda);

    /**
     * \~japanese-en OpenQASMのU3ゲートを作成する。
     * 
     * 具体的なゲートについてはOpenQASMのマニュアルを参照
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @param[in] lambda 回転角の第一引数
     * @param[in] phi 回転角の第二引数
     * @param[in] theta 回転角の第三引数
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate U3 gate of OpenQASM
     * 
     * See the OpenQASM manual for details of gates
     * @param[in] qubit_index Subscript of target qubit
     * @param[in] lambda The first argument of rotation angle
     * @param[in] phi The second argument of rotation angle
     * @param[in] theta The third argument of rotation angle
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* U3(UINT qubit_index, double theta, double phi, double lambda);

    /**
     * \~japanese-en \f$X\f$回転ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @param[in] angle 回転角
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$X\f$ rotation gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @param[in] angle rotation angle
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* RX(UINT qubit_index,double angle);

    /**
     * \~japanese-en \f$Y\f$回転ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @param[in] angle 回転角
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$Y\f$ rotation gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @param[in] angle rotation angle
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* RY(UINT qubit_index, double angle);

    /**
     * \~japanese-en \f$Z\f$回転ゲートを作成する。
     * 
     * @param[in] qubit_index ターゲットとなる量子ビットの添え字
     * @param[in] angle 回転角
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$Z\f$ rotation gate
     * 
     * @param[in] qubit_index Subscript of target qubit
     * @param[in] angle rotation angle
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* RZ(UINT qubit_index, double angle);

    /**
     * \~japanese-en CNOTゲートを作成する
     * 
     * @param[in] control_qubit_index コントロールとなる量子ビットの添え字
     * @param[in] target_qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate CNOT gate
     * 
     * @param[in] control_qubit_index Subscript of control qubit
     * @param[in] target_qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* CNOT(UINT control_qubit_index, UINT target_qubit_index);

    /**
     * \~japanese-en CZゲートを作成する
     * 
     * @param[in] control_qubit_index コントロールとなる量子ビットの添え字
     * @param[in] target_qubit_index ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate CZ gate
     * 
     * @param[in] control_qubit_index Subscript of control qubit
     * @param[in] target_qubit_index Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* CZ(UINT control_qubit_index, UINT target_qubit_index);

    /**
     * \~japanese-en SWAPゲートを作成する
     * 
     * @param[in] qubit_index1 ターゲットとなる量子ビットの添え字
     * @param[in] qubit_index2 ターゲットとなる量子ビットの添え字
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate SWAP gate
     * 
     * @param[in] qubit_index1 Subscript of target qubit
     * @param[in] qubit_index2 Subscript of target qubit
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* SWAP(UINT qubit_index1, UINT qubit_index2);

    /**
     * \~japanese-en \f$n\f$-qubit パウリ演算子のゲートを作成する
     * 
     * 例えば\f$Y_1 X_3\f$であれば、<code>target_qubit_index_list = {1,3}, pauli_id_list = {2,1};</code>である。
     * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字のリスト
     * @param[in] pauli_id_list その量子ビットに作用するパウリ演算子。\f${I,X,Y,Z}\f$が\f${0,1,2,3}\f$に対応する。
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$n\f$-qubit Pauli gate
     * 
     * For example, if \f$Y_1 X_3\f$, then <code>target_qubit_index_list = {1,3}, pauli_id_list = {2,1};</code>.
     * @param[in] target_qubit_index_list List of subscript of target qubit
     * @param[in] pauli_id_list Pauli operator acting on qubit. \f${I, X, Y, Z}\f$ corresponds to \f$ {0,1,2,3} \f$.
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* Pauli(std::vector<UINT> target_qubit_index_list, std::vector<UINT> pauli_id_list);

    /**
     * \~japanese-en \f$n\f$-qubit パウリ演算子の回転ゲートを作成する
     * 
     * 例えば\f$Y_1 X_3\f$であれば、<code>target_qubit_index_list = {1,3}, pauli_id_list = {2,1};</code>である。
     * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字のリスト
     * @param[in] pauli_id_list その量子ビットに作用するパウリ演算子。\f${I,X,Y,Z}\f$が\f${0,1,2,3}\f$に対応する。
     * @param[in] angle 回転角
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate \f$n\f$-qubit Pauli rotation gate
     * 
     * For example, if \f$Y_1 X_3\f$, then <code>target_qubit_index_list = {1,3}, pauli_id_list = {2,1};</code>.
     * @param[in] target_qubit_index_list List of subscript of target qubit
     * @param[in] pauli_id_list Pauli operator acting on qubit. \f${I, X, Y, Z}\f$ corresponds to \f$ {0,1,2,3} \f$.
     * @param[in] angle Rotation angle
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* PauliRotation(std::vector<UINT> target_qubit_index_list, std::vector<UINT> pauli_id_list, double angle);

    /**
     * \~japanese-en \f$n\f$-qubit 行列を用いて1-qubitゲートを生成する。
     * 
     * @param[in] target_qubit_index ターゲットとなる量子ビットの添え字
     * @param[in] matrix 作用するゲートの\f$2\times 2\f$の複素行列
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate 1-qubit gate using \f$n\f$-qubit matrix
     * 
     * @param[in] target_qubit_index Subscript of target qubit
     * @param[in] matrix \F$2\times 2\f$ complex matrix of gates to be operated
     * @return Instance of generated gate
     */
    DllExport QuantumGateMatrix* DenseMatrix(UINT target_qubit_index, ComplexMatrix matrix);

    /**
     * \~japanese-en \f$n\f$-qubit 行列を用いてn-qubitゲートを生成する。
     * 
     * <code>target_qubit_index_list</code>の要素数を\f$m\f$としたとき、<code>matrix</code>は\f$2^m \times 2^m \f$の複素行列でなくてはいけない。
     * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字
     * @param[in] matrix 作用するゲートの複素行列。
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Generate n-qubit gate using \f$n\f$-qubit matrix
     * 
     * When the length of <code> target_qubit_index_list </ code> is \f$m\f$, <code> matrix </ code> has to be a complex matrix of \f$2 ^m \times 2^m\f$ dimension.
     * @param[in] target_qubit_index_list List of subscript of target qubit
     * @param[in] matrix Gate matrix to be operated
     * @return Instance of generated gate
     */
    DllExport QuantumGateMatrix* DenseMatrix(std::vector<UINT> target_qubit_index_list, ComplexMatrix matrix);

	/**
	 * \~japanese-en \f$n\f$-qubit スパースな行列を用いてn-qubitゲートを生成する。
	 *
	 * <code>target_qubit_index_list</code>の要素数を\f$m\f$としたとき、<code>matrix</code>は\f$2^m \times 2^m \f$の複素行列でなくてはいけない。
	 * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字
	 * @param[in] matrix 作用するゲートの複素行列。
	 * @return 作成されたゲートのインスタンス
	 */
	/**
	 * \~english Generate n-qubit gate using \f$n\f$-qubit sparse matrix
	 *
	 * When the length of <code> target_qubit_index_list </ code> is \f$m\f$, <code> matrix </ code> has to be a complex matrix of \f$2 ^m \times 2^m\f$ dimension.
         * @param[in] target_qubit_index_list List of subscript of target qubit
	 * @param[in] matrix Gate matrix to be operated
	 * @return Instance of generated gate
	 */
	DllExport QuantumGateBase* SparseMatrix(std::vector<UINT> target_qubit_index_list, SparseComplexMatrix matrix);

	/**
	 * \~japanese-en \f$n\f$-qubit のランダムユニタリゲートを作成する。
	 *
	 * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字
	 * @return 作成されたゲートのインスタンス
	 */
	/**
	 * \~english Generate n-qubit random unitary gate
	 *
	 * @param[in] target_qubit_index_list List of subscript of target qubit
	 * @return Instance of generated gate
	 */
	DllExport QuantumGateMatrix* RandomUnitary(std::vector<UINT> target_qubit_index_list);

	/**
	 * \~japanese-en \f$n\f$-qubit の可逆古典回路を作用する。
	 *
	 * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字
	 * @param[in] function_ptr 可逆古典回路の動作をする関数
	 * @return 作成されたゲートのインスタンス
	 */
	/**
	 * \~english Operate on \f$n\f$-qubit reversible classical circuit.
	 *
	 * @param[in] target_qubit_index_list List of subscript of target qubit
	 * @param[in] function_ptr Functions that operate reversible classical circuits
	 * @return Instance of generated gate
	 */
	DllExport QuantumGateBase* ReversibleBoolean(std::vector<UINT> target_qubit_index_list, std::function<ITYPE(ITYPE,ITYPE)>);

	/**
	 * \~japanese-en 量子状態に対して量子状態を反射する。
	 *
	 * @param[in] reflection_state 反射に用いられる量子状態
	 * @return 作成されたゲートのインスタンス
	 */
	/**
	 * \~english Reflect a quantum state with respect to quantum state.
	 *
	 * @param[in] reflection_state Quantum states used for reflection
	 * @return Instance of generated gate
	 */
	DllExport QuantumGateBase* StateReflection(const QuantumStateBase* reflection_state);


    /**
     * \~japanese-en bit-flipノイズを発生させるゲート
     *
     * @param[in] target_index ターゲットとなる量子ビットの添え字
     * @param[in] prob エラーが生じる確率
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Gate that generates bit-flip noise
     *
     * @param[in] target_index Subscript of target qubit
     * @param[in] prob Probability of error
     * @return Instance of generated gate
     */

    DllExport QuantumGateBase* BitFlipNoise(UINT target_index, double prob);

    /**
     * \~japanese-en phase-flipノイズを発生させるゲート
     *
     * @param[in] target_index ターゲットとなる量子ビットの添え字
     * @param[in] prob エラーが生じる確率
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Gate that generates phase-flip noise
     *
     * @param[in] target_index Subscript of target qubit
     * @param[in] prob Probability of error
     * @return Instance of generated gate
     */

    DllExport QuantumGateBase* DephasingNoise(UINT target_index, double prob);

    /**
     * \~japanese-en bit-flipとphase-flipを同じ確率でノイズを発生させるゲート
     *
     * @param[in] target_index ターゲットとなる量子ビットの添え字
     * @param[in] prob エラーが生じる確率
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english A gate that generates bit-flip and phase-flip noises with the same probability
     *
     * @param[in] target_index Subscript of target qubit
     * @param[in] prob Probability of error
     * @return Instance of generated gate
     */

    DllExport QuantumGateBase* IndependentXZNoise(UINT target_index, double prob);

    /**
     * \~japanese-en Depolarizin noiseを発生させるゲート
     *
     * X,Y,Zがそれぞれ<code>prob/3</code>の確率で生じる。
     * @param[in] target_index ターゲットとなる量子ビットの添え字
     * @param[in] prob エラーが生じる確率
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english A gate that generates Depolarizin noise
     *
     * X, Y, and Z occur with a probability of <code> prob /3 </code>.
     * @param[in] target_index Subscript of target qubit
     * @param[in] prob Probability of error
     * @return Instance of generated gate
     */

    DllExport QuantumGateBase* DepolarizingNoise(UINT target_index, double prob);

	/**
	* \~japanese-en Two-qubit depolarizin noiseを発生させるゲート
	*
	* IIを除くtwo qubit Pauli operationがそれぞれ<code>prob/15</code>の確率で生じる。
	* @param[in] target_index1 ターゲットとなる量子ビットの添え字
	* @param[in] target_index2 ターゲットとなる量子ビットの添え字
	* @param[in] prob エラーが生じる確率
	* @return 作成されたゲートのインスタンス
	*/

	/**
	* \~english A gate that generates two-qubit depolarizin noise
	*
	* Two qubit Pauli operations except II occur with a probability of <code>prob/15</code>.
	* @param[in] target_index1 Subscript of target qubit
	* @param[in] target_index2 Subscript of target qubit
	* @param[in] prob Probability of error
        * @return Instance of generated gate
	*/
	
	
	DllExport QuantumGateBase* TwoQubitDepolarizingNoise(UINT target_index1, UINT target_index2, double prob);

	/**
	 * \~japanese-en Amplitude damping noiseを発生させるゲート
	 *
	 * @param[in] target_index ターゲットとなる量子ビットの添え字
	 * @param[in] prob エラーが生じる確率
	 * @return 作成されたゲートのインスタンス
	 */
       /**
         * \~english A gate that generates Amplitude damping noise
         *
         * @param[in] target_index Subscript of target qubit
         * @param[in] prob Probability of error
         * @return Instance of generated gate
         */
	DllExport QuantumGateBase* AmplitudeDampingNoise(UINT target_index, double prob);

    /**
     * \~japanese-en 測定を行う
     *
     * @param[in] target_index ターゲットとなる量子ビットの添え字
     * @param[in] classical_register_address 測定値を格納する古典レジスタの場所
     * @return 作成されたゲートのインスタンス
     */
    /**
     * \~english Do measurement
     *
     * @param[in] target_index Subscript of target qubit
     * @param[in] classical_register_address Location of classic register to store measurement values
     * @return Instance of generated gate
     */
    DllExport QuantumGateBase* Measurement(UINT target_index, UINT classical_register_address);


}

