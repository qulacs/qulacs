#pragma once
/**
 * ゲートを生成するための関数
 *
 * @file gate_factory.hpp
 */

#include <string>
#include <vector>

#include "gate.hpp"
#include "gate_general.hpp"
#include "gate_matrix_diagonal.hpp"
#include "gate_matrix_sparse.hpp"
#include "gate_named_npair.hpp"
#include "gate_named_one.hpp"
#include "gate_named_two.hpp"
#include "gate_noisy_evolution.hpp"
#include "gate_reflect.hpp"
#include "gate_reversible.hpp"
#include "observable.hpp"
#include "type.hpp"
namespace gate {
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
 * FusedSWAP    :   FusedSWAP \<target1\> \<target2\> \<blocksize\>
 * U1           :   U1 \<index\> \<angle1\>
 * U2           :   U2 \<index\> \<angle1\> \<angle2\>
 * U3           :   U3 \<index\> \<angle1\> \<angle2\> \<angle3\>
 * RX           :   RX \<index\> \<angle1\>
 * RY           :   RY \<index\> \<angle1\>
 * RZ           :   RZ \<index\> \<angle1\>
 * DifRot X     :   RDX \<index\>
 * DifRot Y     :   RDY \<index\>
 * DifRot Z     :   RDZ \<index\>
 * MultiRot     :   RM \<paulistr\> \<index1\> \<index2\> ... \<theta\> (for
 * どれにも合致しない場合はNULLを返す
 * example: "RM XYZ 2 3 1 0.123") DifMultiRot  :   RDM \<paulistr\> \<index1\>
 * \<index2\> ...  (for example: "RDM XYZ 2 3 1") general U    :   U
 * \<index_count\> \<index1\> \<index2\> ... \<element1_real\> \<element1_imag\>
 * \<element2_real\> ...
 * @param[in] gate_string ゲートを生成する文字列
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGateBase* create_quantum_gate_from_string(
    std::string gate_string);

/**
 * \~japanese-en Identityゲートを作成する。
 *
 * 作用しても状態は変わらないが、ノイズなどが付与された際の挙動が異なる。
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* Identity(UINT qubit_index);

/**
 * \~japanese-en \f$X\f$ゲートを作成する。
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* X(UINT qubit_index);

/**
 * \~japanese-en \f$Y\f$ゲートを作成する。
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* Y(UINT qubit_index);

/**
 * \~japanese-en \f$Z\f$ゲートを作成する。
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* Z(UINT qubit_index);

/**
 * \~japanese-en Hadamardゲートを作成する。
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* H(UINT qubit_index);

/**
 * \~japanese-en \f$S\f$ゲートを作成する。
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* S(UINT qubit_index);

/**
 * \~japanese-en \f$S^{\dagger}\f$ゲートを作成する。
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* Sdag(UINT qubit_index);

/**
 * \~japanese-en \f$T\f$ゲートを作成する。
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* T(UINT qubit_index);

/**
 * \~japanese-en \f$T^{\dagger}\f$ゲートを作成する。
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* Tdag(UINT qubit_index);

/**
 * \~japanese-en \f$\sqrt{X}\f$ゲートを作成する。
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* sqrtX(UINT qubit_index);

/**
 * \~japanese-en \f$\sqrt{X}^{\dagger}\f$ゲートを作成する。
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* sqrtXdag(UINT qubit_index);

/**
 * \~japanese-en \f$\sqrt{Y}\f$ゲートを作成する。
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* sqrtY(UINT qubit_index);

/**
 * \~japanese-en \f$\sqrt{Y}^{\dagger}\f$ゲートを作成する。
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* sqrtYdag(UINT qubit_index);

/**
 * \~japanese-en <code>qubit_index</code>を0へ射影するゲートを作成する
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* P0(UINT qubit_index);

/**
 * \~japanese-en <code>qubit_index</code>を1へ射影するゲートを作成する
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitGate* P1(UINT qubit_index);

/**
 * \~japanese-en OpenQASMのU1ゲートを作成する。
 *
 * 具体的なゲートについてはOpenQASMのマニュアルを参照
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @param[in] lambda 回転角の第一引数
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGateMatrix* U1(UINT qubit_index, double lambda);

/**
 * \~japanese-en OpenQASMのU2ゲートを作成する。
 *
 * 具体的なゲートについてはOpenQASMのマニュアルを参照
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @param[in] lambda 回転角の第一引数
 * @param[in] phi 回転角の第二引数
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGateMatrix* U2(UINT qubit_index, double phi, double lambda);

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
DllExport QuantumGateMatrix* U3(
    UINT qubit_index, double theta, double phi, double lambda);

/**
 * \~japanese-en \f$X\f$回転ゲートを作成する。
 *
 * @par Matrix Representation
 *
 * @f[
 * R_X(\theta) = \exp(i\frac{\theta}{2} X) =
 *     \begin{pmatrix}
 *     \cos(\frac{\theta}{2})  & i\sin(\frac{\theta}{2}) \\
 *     i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
 *     \end{pmatrix}
 * @f]
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @param[in] angle 回転角
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitRotationGate* RX(UINT qubit_index, double angle);

/**
 * \~japanese-en \f$Y\f$回転ゲートを作成する。
 *
 * @par Matrix Representation
 *
 * @f[
 * R_Y(\theta) = \exp(i\frac{\theta}{2} Y) =
 *     \begin{pmatrix}
 *     \cos(\frac{\theta}{2})  & \sin(\frac{\theta}{2}) \\
 *     -\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
 *     \end{pmatrix}
 * @f]
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @param[in] angle 回転角
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitRotationGate* RY(UINT qubit_index, double angle);

/**
 * \~japanese-en \f$Z\f$回転ゲートを作成する。
 *
 * @par Matrix Representation
 *
 * @f[
 * R_Z(\theta) = \exp(i\frac{\theta}{2} Z) =
 *     \begin{pmatrix}
 *     e^{i\frac{\theta}{2}} & 0 \\
 *     0 & e^{-i\frac{\theta}{2}}
 *     \end{pmatrix}
 * @f]
 *
 * @param[in] qubit_index ターゲットとなる量子ビットの添え字
 * @param[in] angle 回転角
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneQubitRotationGate* RZ(UINT qubit_index, double angle);

DllExport ClsOneQubitRotationGate* RotInvX(UINT qubit_index, double angle);
DllExport ClsOneQubitRotationGate* RotInvY(UINT qubit_index, double angle);
DllExport ClsOneQubitRotationGate* RotInvZ(UINT qubit_index, double angle);
DllExport ClsOneQubitRotationGate* RotX(UINT qubit_index, double angle);
DllExport ClsOneQubitRotationGate* RotY(UINT qubit_index, double angle);
DllExport ClsOneQubitRotationGate* RotZ(UINT qubit_index, double angle);
/**
 * \~japanese-en CNOTゲートを作成する
 *
 * @param[in] control_qubit_index コントロールとなる量子ビットの添え字
 * @param[in] target_qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneControlOneTargetGate* CNOT(
    UINT control_qubit_index, UINT target_qubit_index);

/**
 * \~japanese-en CZゲートを作成する
 *
 * @param[in] control_qubit_index コントロールとなる量子ビットの添え字
 * @param[in] target_qubit_index ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsOneControlOneTargetGate* CZ(
    UINT control_qubit_index, UINT target_qubit_index);

/**
 * \~japanese-en SWAPゲートを作成する
 *
 * @param[in] qubit_index1 ターゲットとなる量子ビットの添え字
 * @param[in] qubit_index2 ターゲットとなる量子ビットの添え字
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsTwoQubitGate* SWAP(UINT qubit_index1, UINT qubit_index2);

/**
 * \~japanese-en FusedSWAPゲートを作成する
 *
 * @param[in] qubit_index1 ターゲットとなる量子ビットの添え字
 * @param[in] qubit_index2 ターゲットとなる量子ビットの添え字
 * @param[in] block_size ターゲットとなる量子ブロックサイズ
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsNpairQubitGate* FusedSWAP(
    UINT qubit_index1, UINT qubit_index2, UINT block_size);

/**
 * \f$n\f$-qubit パウリ演算子のゲートを作成する
 *
 * 例えば\f$Y_1 X_3\f$であれば、<code>target_qubit_index_list = {1,3},
 * pauli_id_list = {2,1};</code>である。
 * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字のリスト
 * @param[in] pauli_id_list
 * その量子ビットに作用するパウリ演算子。\f${I,X,Y,Z}\f$が\f${0,1,2,3}\f$に対応する。
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsPauliGate* Pauli(
    std::vector<UINT> target_qubit_index_list, std::vector<UINT> pauli_id_list);

/**
 * \f$n\f$-qubit パウリ演算子の回転ゲートを作成する
 *
 * 例えば\f$Y_1 X_3\f$であれば、<code>target_qubit_index_list = {1,3},
 * pauli_id_list = {2,1};</code>である。
 * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字のリスト
 * @param[in] pauli_id_list
 * その量子ビットに作用するパウリ演算子。\f${I,X,Y,Z}\f$が\f${0,1,2,3}\f$に対応する。
 * @param[in] angle 回転角
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsPauliRotationGate* PauliRotation(
    std::vector<UINT> target_qubit_index_list, std::vector<UINT> pauli_id_list,
    double angle);

/**
 * \f$n\f$-qubit 行列を用いて1-qubitゲートを生成する。
 *
 * @param[in] target_qubit_index ターゲットとなる量子ビットの添え字
 * @param[in] matrix 作用するゲートの\f$2\times 2\f$の複素行列
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGateMatrix* DenseMatrix(
    UINT target_qubit_index, ComplexMatrix matrix);

/**
 * \f$n\f$-qubit 行列を用いてn-qubitゲートを生成する。
 *
 * <code>target_qubit_index_list</code>の要素数を\f$m\f$としたとき、<code>matrix</code>は\f$2^m
 * \times 2^m \f$の複素行列でなくてはいけない。
 * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字
 * @param[in] matrix 作用するゲートの複素行列。
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGateMatrix* DenseMatrix(
    std::vector<UINT> target_qubit_index_list, ComplexMatrix matrix);

/**
 * \f$n\f$-qubit スパースな行列を用いてn-qubitゲートを生成する。
 *
 * <code>target_qubit_index_list</code>の要素数を\f$m\f$としたとき、<code>matrix</code>は\f$2^m
 * \times 2^m \f$の複素行列でなくてはいけない。
 * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字
 * @param[in] matrix 作用するゲートの複素行列。
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGateSparseMatrix* SparseMatrix(
    std::vector<UINT> target_qubit_index_list, SparseComplexMatrix matrix);

/**
 * \f$n\f$-qubit 対角な行列を用いてn-qubitゲートを生成する。
 *
 * <code>target_qubit_index_list</code>の要素数を\f$m\f$としたとき、<code>matrix</code>は\f$2^m
 * \times 2^m \f$の複素行列でなくてはいけない。
 * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字
 * @param[in] matrix 作用するゲートの複素行列の対角成分。
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGateDiagonalMatrix* DiagonalMatrix(
    std::vector<UINT> target_qubit_index_list, ComplexVector diagonal_element);

/**
 * \f$n\f$-qubit のランダムユニタリゲートを作成する。
 *
 * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字
 * @param[in] seed 乱数のシード値
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGateMatrix* RandomUnitary(
    std::vector<UINT> target_qubit_index_list);
DllExport QuantumGateMatrix* RandomUnitary(
    std::vector<UINT> target_qubit_index_list, UINT seed);

/**
 * \f$n\f$-qubit の可逆古典回路を作用する。
 *
 * @param[in] target_qubit_index_list ターゲットとなる量子ビットの添え字
 * @param[in] function_ptr 可逆古典回路の動作をする関数
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsReversibleBooleanGate* ReversibleBoolean(
    std::vector<UINT> target_qubit_index_list,
    std::function<ITYPE(ITYPE, ITYPE)>);

/**
 * 量子状態に対して量子状態を反射する。
 *
 * @param[in] reflection_state 反射に用いられる量子状態
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsStateReflectionGate* StateReflection(
    const QuantumState* reflection_state);

/**
 * 量子ゲートの線型結合を作成する．
 *
 * @param[in] coeffs 係数のリスト
 * @param[in] gate_list ゲートのリスト
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_LinearCombination* LinearCombination(
    const std::vector<CPPCTYPE>& coefs,
    const std::vector<QuantumGateBase*>& gate_list);

/**
 * bit-flipノイズを発生させるゲート
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] prob エラーが生じる確率
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Probabilistic* BitFlipNoise(
    UINT target_index, double prob);
/**
 * bit-flipノイズを発生させるゲート
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] prob エラーが生じる確率
 * @param[in] seed 乱数のシード値
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Probabilistic* BitFlipNoise(
    UINT target_index, double prob, UINT seed);

/**
 * phase-flipノイズを発生させるゲート
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] prob エラーが生じる確率
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Probabilistic* DephasingNoise(
    UINT target_index, double prob);
/**
 * phase-flipノイズを発生させるゲート
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] prob エラーが生じる確率
 * @param[in] seed 乱数のシード値
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Probabilistic* DephasingNoise(
    UINT target_index, double prob, UINT seed);

/**
 * bit-flipとphase-flipを同じ確率でノイズを発生させるゲート
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] prob エラーが生じる確率
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Probabilistic* IndependentXZNoise(
    UINT target_index, double prob);
/**
 * bit-flipとphase-flipを同じ確率でノイズを発生させるゲート
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] prob エラーが生じる確率
 * @param[in] seed 乱数のシード値
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Probabilistic* IndependentXZNoise(
    UINT target_index, double prob, UINT seed);

/**
 * Depolarizin noiseを発生させるゲート
 *
 * X,Y,Zがそれぞれ<code>prob/3</code>の確率で生じる。
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] prob エラーが生じる確率
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Probabilistic* DepolarizingNoise(
    UINT target_index, double prob);
/**
 * Depolarizin noiseを発生させるゲート
 *
 * X,Y,Zがそれぞれ<code>prob/3</code>の確率で生じる。
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] prob エラーが生じる確率
 * @param[in] seed 乱数のシード値
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Probabilistic* DepolarizingNoise(
    UINT target_index, double prob, UINT seed);

/**
 * Two-qubit depolarizin noiseを発生させるゲート
 *
 * IIを除くtwo qubit Pauli
 * operationがそれぞれ<code>prob/15</code>の確率で生じる。
 * @param[in] target_index1 ターゲットとなる量子ビットの添え字
 * @param[in] target_index2 ターゲットとなる量子ビットの添え字
 * @param[in] prob エラーが生じる確率
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Probabilistic* TwoQubitDepolarizingNoise(
    UINT target_index1, UINT target_index2, double prob);
/**
 * Two-qubit depolarizin noiseを発生させるゲート
 *
 * IIを除くtwo qubit Pauli
 * operationがそれぞれ<code>prob/15</code>の確率で生じる。
 * @param[in] target_index1 ターゲットとなる量子ビットの添え字
 * @param[in] target_index2 ターゲットとなる量子ビットの添え字
 * @param[in] prob エラーが生じる確率
 * @param[in] seed 乱数のシード値
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Probabilistic* TwoQubitDepolarizingNoise(
    UINT target_index1, UINT target_index2, double prob, UINT seed);

/**
 * Amplitude damping noiseを発生させるゲート
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] prob エラーが生じる確率
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_CPTP* AmplitudeDampingNoise(
    UINT target_index, double prob);
/**
 * Amplitude damping noiseを発生させるゲート
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] prob エラーが生じる確率
 * @param[in] seed 乱数のシード値
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_CPTP* AmplitudeDampingNoise(
    UINT target_index, double prob, UINT seed);

/**
 * 測定を行う
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] classical_register_address 測定値を格納する古典レジスタの場所
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Instrument* Measurement(
    UINT target_index, UINT classical_register_address);
/**
 * 測定を行う
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] classical_register_address 測定値を格納する古典レジスタの場所
 * @param[in] seed 乱数のシード値
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Instrument* Measurement(
    UINT target_index, UINT classical_register_address, UINT seed);
/**
 * 測定を行う
 *
 * @param[in] pauli 測定を行うパウリ演算子
 * @param[in] classical_register_address 測定値を格納する古典レジスタの場所
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Instrument* MultiQubitPauliMeasurement(
    const std::vector<UINT>& target_index_list,
    const std::vector<UINT>& pauli_id_list, UINT classical_register_address);
/**
 * 測定を行う
 *
 * @param[in] pauli 測定を行うパウリ演算子
 * @param[in] classical_register_address 測定値を格納する古典レジスタの場所
 * @param[in] seed 乱数のシード値
 * @return 作成されたゲートのインスタンス
 */
DllExport QuantumGate_Instrument* MultiQubitPauliMeasurement(
    const std::vector<UINT>& target_index_list,
    const std::vector<UINT>& pauli_id_list, UINT classical_register_address,
    UINT seed);

/**
 * Noisy Evolution
 * TODO: do this comment
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] classical_register_address 測定値を格納する古典レジスタの場所
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsNoisyEvolution* NoisyEvolution(Observable* hamiltonian,
    std::vector<GeneralQuantumOperator*> c_ops, double time, double dt = 1e-6);

/**
 * Noisy Evolution
 * TODO: do this comment
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] classical_register_address 測定値を格納する古典レジスタの場所
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsNoisyEvolution_fast* NoisyEvolution_fast(Observable* hamiltonian,
    std::vector<GeneralQuantumOperator*> c_ops, double time);

/**
 * Noisy Evolution
 * TODO: do this comment
 *
 * @param[in] target_index ターゲットとなる量子ビットの添え字
 * @param[in] classical_register_address 測定値を格納する古典レジスタの場所
 * @return 作成されたゲートのインスタンス
 */
DllExport ClsNoisyEvolution_auto* NoisyEvolution_auto(Observable* hamiltonian,
    std::vector<GeneralQuantumOperator*> c_ops, double time);

/**
 * \~japanese-en ptreeからゲートを構築
 */
DllExport QuantumGateBase* from_ptree(const boost::property_tree::ptree& pt);
}  // namespace gate
