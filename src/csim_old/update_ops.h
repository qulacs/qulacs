/*
 rules for arguments of update functions:
   The order of arguments must be 
      information_about_applying_qubits -> Pauli_operator -> information_about_applying_gate_matrix_elements -> a_kind_of_rotation_angle -> state_vector -> dimension
   If there is control-qubit and target-qubit, control-qubit is the first argument.
   If an array of which the size is not known comes, the size of that array follows.
  
  Definition of update function is divided to named_gates, single_target_qubit_gates, multiple_target_qubit_gates, QFT_gates
 */


#pragma once

#include "type.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/** X gate **/
 
 /**
 * \~english
 * Apply the Pauli X gate to the quantum state.
 *
 * Apply the Pauli X gate to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * パウリX演算を作用させて状態を更新。
 * 
 * パウリX演算を作用させて状態を更新。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void X_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply the Pauli Y gate to the quantum state.
 *
 * Apply the Pauli Y gate to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * パウリY演算を作用させて状態を更新。
 * 
 * パウリY演算を作用させて状態を更新。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void Y_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply the Pauli Z gate to the quantum state.
 *
 * Apply the Pauli Z gate to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * パウリZ演算を作用させて状態を更新。
 * 
 * パウリZ演算を作用させて状態を更新。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void Z_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply S gate to the quantum state.
 *
 * Apply S gate to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * 位相演算 S を作用させて状態を更新。
 * 
 * 位相演算 S = diag(1,i) を作用させて状態を更新。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void S_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply S gate to the quantum state.
 *
 * Apply S gate to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * 位相演算 S^dag を作用させて状態を更新。
 * 
 * 位相演算 S^dag = diag(1,-i) を作用させて状態を更新。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void Sdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply T gate to the quantum state.
 *
 * Apply T gate to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * T 演算を作用させて状態を更新。
 * 
 * T 演算（pi/8演算）、 T = diag(1,exp(i pi/4)) を作用させて状態を更新。非クリフォード演算。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void T_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply T^dag gate to the quantum state.
 *
 * Apply T^dag gate to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * T^dag 演算を作用させて状態を更新。
 * 
 * T 演算のエルミート共役、 T^dag = diag(1,exp(-i pi/4)) を作用させて状態を更新。非クリフォード演算。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
 DllExport void Tdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply the square root of the X gate to the quantum state.
 *
 * Apply the square root of the X gate to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * パウリ X 演算子の平方根の演算子を作用させて状態を更新。
 * 
 * パウリ X 演算子の平方根の演算子を作用させて状態を更新。非クリフォード演算。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void sqrtX_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply hermitian conjugate of the square root of the X gate to the quantum state.
 *
 * Apply hermitian conjugate of the square root of the X gate to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * パウリ X 演算子の平方根に対してエルミート共役な演算子を作用させて状態を更新。
 * 
 * パウリ X 演算子の平方根に対してエルミート共役な演算子を作用させて状態を更新。非クリフォード演算。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void sqrtXdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply the square root of the Y gate to the quantum state.
 *
 * Apply the square root of the Y gate to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * パウリ Y 演算子の平方根の演算子を作用させて状態を更新。
 * 
 * パウリ Y 演算子の平方根の演算子を作用させて状態を更新。非クリフォード演算。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void sqrtY_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply hermitian conjugate of the square root of the Y gate to the quantum state.
 *
 * Apply hermitian conjugate of the square root of the Y gate to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * パウリ Y 演算子の平方根に対してエルミート共役な演算子を作用させて状態を更新。
 * 
 * パウリ Y 演算子の平方根に対してエルミート共役な演算子を作用させて状態を更新。非クリフォード演算。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void sqrtYdag_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply the Hadamard gate to the quantum state.
 *
 * Apply the Hadamard gate to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * アダマール演算子を作用させて状態を更新。
 * 
 * アダマール演算子を作用させて状態を更新。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void H_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim);

/** Hadamard gate multiplied sqrt(2) **/
//DllExport void H_gate_unnormalized(UINT target_qubit_index, CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply the CNOT gate to the quantum state.
 * 
 * Apply the CNOT gate to the quantum state.
 * @param[in] control_qubit_index index of control qubit
 * @param[in] target_qubit_index index of target qubit
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 * \~japanese-en
 * CNOT演算を作用させて状態を更新。
 *
 * 2量子ビット演算、CNOT演算を作用させて状態を更新。
 * @param[in] control_qubit_index 制御量子ビットのインデックス
 * @param[in] target_qubit_index ターゲット量子ビットのインデックス
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 */
DllExport void CNOT_gate(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply the CZ gate to the quantum state.
 * 
 * Apply the CZ gate to the quantum state.
 * @param[in] control_qubit_index index of control qubit
 * @param[in] target_qubit_index index of target qubit
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 * \~japanese-en
 * CZ演算を作用させて状態を更新。
 *
 * 2量子ビット演算、CZ演算を作用させて状態を更新。制御量子ビットとターゲット量子ビットに対して対称に作用（インデックスを入れ替えても同じ作用）。
 * @param[in] control_qubit_index 制御量子ビットのインデックス
 * @param[in] target_qubit_index ターゲット量子ビットのインデックス
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 */
DllExport void CZ_gate(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply the SWAP to the quantum state.
 * 
 * Apply the SWAP to the quantum state.
 * @param[in] control_qubit_index index of control qubit
 * @param[in] target_qubit_index index of target qubit
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 * \~japanese-en
 * SWAP演算を作用させて状態を更新。
 *
 * 2量子ビット演算、SWAP演算を作用させて状態を更新。２つの量子ビットに対して対称に作用する（インデックスを入れ替えても同じ作用）。
 * @param[in] target_qubit_index_0 作用する量子ビットのインデックス
 * @param[in] target_qubit_index_1 作用する量子ビットのインデックス
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 */
DllExport void SWAP_gate(UINT target_qubit_index_0, UINT target_qubit_index_1, CTYPE *state, ITYPE dim);

/**
 * \~english
 * Project the quantum state to the 0 state.
 * 
 * Project the quantum state to the 0 state. The output state is not normalized.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 * \~japanese-en
 * 0状態への射影
 *
 * 0状態への射影演算子 |0><0| を作用させて状態を更新する。ノルムは規格化されない。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void P0_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim);

/**
 * \~english
 * Project the quantum state to the 1 state.
 * 
 * Project the quantum state to the 1 state. The output state is not normalized.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 * \~japanese-en
 * 1状態への射影
 *
 * 1状態への射影演算子 |1><1| を作用させて状態を更新する。ノルムは規格化されない。
 * @param[in] target_qubit_index 作用する量子ビットのインデックス
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void P1_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim);

/**
 * \~english
 * Normalize the quantum state.
 * 
 * Normalize the quantum state by multiplying the normalization factor 1/sqrt(norm).
 * @param[in] norm norm of the state
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 * \~japanese-en
 * 状態を規格化
 *
 * 状態に対して 1/sqrt(norm) 倍をする。norm が量子状態のノルムである場合にはノルムが1になるように規格化される。
 * @param[in] norm ノルム
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 */
DllExport void normalize(double norm, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply a X rotation gate by angle to the quantum state.
 *
 * Apply a X rotation gate by angle to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in] angle angle of the rotation
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * X軸の回転演算を作用させて状態を更新
 *
 * X軸の回転演算
 * 
 * cos (angle/2) I + i sin (angle/2) X
 *
 * を作用させて状態を更新。angleは回転角。
 * @param[in] target_qubit_index 量子ビットのインデックス
 * @param[in] angle 回転角
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void RX_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply a Y rotation gate by angle to the quantum state.
 *
 * Apply a Y rotation gate by angle to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in] angle angle of the rotation
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * Y軸の回転演算を作用させて状態を更新
 *
 * Y軸の回転演算
 * 
 * cos (angle/2) I + i sin (angle/2) Y
 *
 * を作用させて状態を更新。angleは回転角。
 * @param[in] target_qubit_index 量子ビットのインデックス
 * @param[in] angle 回転角
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void RY_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply a Z rotation gate by angle to the quantum state.
 *
 * Apply a Z rotation gate by angle to the quantum state.
 * @param[in] target_qubit_index index of the qubit 
 * @param[in] angle angle of the rotation
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * Z軸の回転演算を作用させて状態を更新
 *
 * Z軸の回転演算
 * 
 * cos (angle/2) I + i sin (angle/2) Z
 *
 * を作用させて状態を更新。angleは回転角。
 * @param[in] target_qubit_index 量子ビットのインデックス
 * @param[in] angle 回転角
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void RZ_gate(UINT target_qubit_index, double angle, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply a single-qubit Pauli operator to the quantum state.
 *
 * Apply a single-qubit Pauli operator to the quantum state. Pauli_operator_type must be 0,1,2,3 corresponding to the Pauli I, X, Y, Z operators respectively.
 * 
 *
 * @param[in] target_qubit_index index of the qubit
 * @param[in] Pauli_operator_type type of the Pauli operator 0,1,2,3
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * パウリ演算子を作用させて状態を更新
 *
 * パウリ演算子を作用させて状態を更新。Pauli_operator_type はパウリ演算子 I, X, Y, Z に対応して 0,1,2,3 を指定。
 * @param[in] target_qubit_index 量子ビットのインデックス
 * @param[in] Pauli_operator_type 作用するパウリ演算子のタイプ、0,1,2,3。それぞれI, X, Y, Zに対応。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 * 
 * 
 */
DllExport void single_qubit_Pauli_gate(UINT target_qubit_index, UINT Pauli_operator_type, CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply a single-qubit Pauli rotation operator to the quantum state.
 *
 * Apply a single-qubit Pauli rotation operator to the quantum state. Pauli_operator_type must be 0,1,2,3 corresponding to the Pauli I, X, Y, Z operators respectively.
 * 
 *
 * @param[in] target_qubit_index index of the qubit
 * @param[in] Pauli_operator_type type of the Pauli operator 0,1,2,3
 * @param[in] angle rotation angle
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * パウリ演算子の回転演算を作用させて状態を更新
 *
 * パウリ演算子 A = I, X, Y, Zに対する回転角 angle の回転演算
 *
 * cos (angle/2) I + i sin (angle/2) A
 *
 * を作用させて状態を更新。Pauli_operator_type はパウリ演算子 I, X, Y, Z に対応して 0,1,2,3 を指定。
 * @param[in] target_qubit_index 量子ビットのインデックス
 * @param[in] Pauli_operator_type 作用するパウリ演算子のタイプ、0,1,2,3。それぞれI, X, Y, Zに対応。
 * @param[in] angle 回転角
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 * 
 * 
 */
DllExport void single_qubit_Pauli_rotation_gate(UINT target_qubit_index, UINT Pauli_operator_index, double angle, CTYPE *state, ITYPE dim);


/**
 * \~english
 * Apply a single-qubit dense operator to the quantum state.
 *
 * Apply a single-qubit dense operator to the quantum state.
 * 
 * @param[in] target_qubit_index index of the qubit
 * @param[in] matrix[4] description of the dense matrix as one-dimensional array
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * 任意の１量子ビット演算を作用させて状態を更新
 *
 * 任意の１量子ビット演算を作用させて状態を更新。１量子ビット演算は、4つの要素をもつ１次元配列 matrix[4] によって指定される。
 *
 * 例）パウリX演算子：{0, 1, 1, 0}、アダマール演算子：{1/sqrt(2),1/sqrt(2),1/sqrt(2),-1/sqrt(2)}。
 * 
 * @param[in] target_qubit_index 量子ビットのインデックス
 * @param[in] matrix[4] １量子ビット演算を指定する4次元配列
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 * 
 */
DllExport void single_qubit_dense_matrix_gate(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply a single-qubit diagonal operator to the quantum state.
 *
 * Apply a single-qubit diagonal operator to the quantum state.
 * 
 * @param[in] target_qubit_index index of the qubit
 * @param[in] diagonal_matrix[2] description of the single-qubit diagonal elements
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * １量子ビットの対角演算を作用させて状態を更新
 *
 * １量子ビットの対角演算を作用させて状態を更新。１量子ビットの対角演算は、その対角成分を定義する2つの要素から成る１次元配列 diagonal_matrix[2] によって指定される。
 *
 * @param[in] target_qubit_index 量子ビットのインデックス
 * @param[in] diagonal_matrix[2] ２つの対角成分を定義する１次元配列
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 * 
 */
DllExport void single_qubit_diagonal_matrix_gate(UINT target_qubit_index, const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply a single-qubit phase operator to the quantum state.
 *
 * Apply a single-qubit phase operator, diag(1,phsae), to the quantum state.
 * 
 * @param[in] target_qubit_index index of the qubit
 * @param[in] phase phase factor
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * １量子ビットの一般位相演算を作用させて状態を更新
 *
 * １量子ビットの一般位相演算 diag(1,phase) を作用させて状態を更新。|1>状態が phase 倍される。
 *
 * @param[in] target_qubit_index 量子ビットのインデックス
 * @param[in] phase 位相因子
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 * 
 */
DllExport void single_qubit_phase_gate(UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply a single-qubit controlled single-qubit gate.
 *
 * Apply a single-qubit controlled single-qubit gate.
 * 
 * @param[in] control_qubit_index index of the control qubit
 * @param[in] control_value value of the control qubit
 * @param[in] target_qubit_index index of the target qubit
 * @param[in] matrix[4] description of the single-qubit dense matrix
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * 単一量子ビットを制御量子ビットとする単一量子ビット演算
 *
 * 単一量子ビットを制御量子ビットとする単一量子ビット演算。制御量子ビット |0> もしくは |1>のどちらの場合に作用するかは control_value = 0,1 によって指定。作用する単一量子ビット演算は matrix[4] によって１次元配列として定義。
 * 
 * @param[in] control_qubit_index 制御量子ビットのインデックス
 * @param[in] control_value 制御量子ビットの値
 * @param[in] target_qubit_index ターゲット量子ビットのインデックス
 * @param[in] matrix[4] 単一量子ビットの記述を与える１次元配列
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *  
 */
DllExport void single_qubit_control_single_qubit_dense_matrix_gate(UINT control_qubit_index, UINT control_value, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply a multi-qubit controlled single-qubit gate.
 *
 * Apply a multi-qubit controlled single-qubit gate.
 * 
 * @param[in] control_qubit_index_list list of the indexes of the control qubits
 * @param[in] control_value_list list of the vlues of the control qubits
 * @param[in] control_qubit_index_count the number of the control qubits
 * @param[in] target_qubit_index index of the target qubit
 * @param[in] matrix[4] description of the single-qubit dense matrix
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * 複数量子ビットを制御量子ビットとする単一量子ビット演算
 *
 * 複数量子ビットを制御量子ビットとする単一量子ビット演算。control_qubit_index_count 個の制御量子ビットについて、どの状態の場合に制御演算が作用するかは control_value_list によって指定。作用する単一量子ビット演算は matrix[4] によって１次元配列として定義。
 * 
 * @param[in] control_qubit_index_list 制御量子ビットのインデックスのリスト
 * @param[in] control_value_list 制御演算が作用する制御量子ビットの値のリスト
 * @param[in] control_qubit_index_count 制御量子ビットの数
 * @param[in] target_qubit_index ターゲット量子ビットのインデックス
 * @param[in] matrix[4] 単一量子ビットの記述を与える１次元配列
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *   
 */
DllExport void multi_qubit_control_single_qubit_dense_matrix_gate(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply multi-qubit Pauli operator to the quantum state with a whole list.
 *
 * Apply multi-qubit Pauli operator to the quantum state with a whole list of the Pauli operators. Pauli_operator_type_list must be a list of n single Pauli operators.
 * 
 * @param[in] Pauli_operator_type_list list of {0,1,2,3} corresponding to I, X, Y, Z for all qubits
 * @param[in] qubit_count the number of the qubits 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * すべての量子ビットに多体のパウリ演算子を作用させて状態を更新
 *
 * 全ての量子ビットに対するパウリ演算子を与えて、多体のパウリ演算子を作用させて状態を更新。Pauli_operator_type_list には全ての量子ビットに対するパウリ演算子を指定。
 * 
 * 例）５量子ビット系の３つめの量子ビットへのX演算、４つめの量子ビットへのZ演算、IZXII：{0,0,1,3,0}（量子ビットの順番は右から数えている）
 * 
 * @param[in] Pauli_operator_type_list 長さ qubit_count の {0,1,2,3} のリスト。 0,1,2,3 はそれぞれパウリ演算子 I, X, Y, Z に対応。
 * @param[in] qubit_count 量子ビットの数
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *   
 */
DllExport void multi_qubit_Pauli_gate_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, CTYPE* state, ITYPE dim_);

/**
 * \~english
 * Apply multi-qubit Pauli operator to the quantum state with a partial list.
 *
 * Apply multi-qubit Pauli operator to the quantum state with a partial list of the Pauli operators. Pauli_operator_type_list must be a list of n single Pauli operators.
 * 
 * @param[in] target_qubit_index_list list of the target qubits
 * @param[in] Pauli_operator_type_list list of {0,1,2,3} corresponding to I, X, Y, Z for the target qubits
 * @param[in] target_qubit_index_count the number of the target qubits 
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * すべての量子ビットに多体のパウリ演算子を作用させて状態を更新
 *
 * 全ての量子ビットに対するパウリ演算子を与えて、多体のパウリ演算子を作用させて状態を更新。Pauli_operator_type_list には全ての量子ビットに対するパウリ演算子を指定。
 * 
 * 例）５量子ビット系の３つめの量子ビットへのX演算、４つめの量子ビットへのZ演算、IZXII：Pauli_operator_type_list ={2,3}, Pauli_operator_type_list ={1,3}（量子ビットの順番は右から数えている）.
 * 
 * @param[in] target_qubit_index_list 作用する量子ビットのインデックスのリスト。
 * @param[in] Pauli_operator_type_list 作用する量子ビットのみに対するパウリ演算子を指定する、長さ target_qubit_index_count の{0,1,2,3}のリスト。0,1,2,3 はそれぞれパウリ演算子 I, X, Y, Z に対応。
 * @param[in] target_qubit_index_count 作用する量子ビットの数
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *   
 */
 DllExport void multi_qubit_Pauli_gate_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply multi-qubit Pauli rotation operator to the quantum state with a whole list.
 *
 * Apply multi-qubit Pauli rotation operator to state with a whole list of the Pauli operators. Pauli_operator_type_list must be a list of n single Pauli operators.
 * Update a quantum state with a mutliple qubits Pauli rotation,
 * cos (angle/2 ) I + i sin ( angle/2 ) A,
 * where A is the Pauli operator specified by Pauli_operator.
 *
 * @param[in] Pauli_operator_type_list array of {0,1,2,3} of the length n_qubits. 0,1,2,3 corresponds to i,x,y,z
 * @param[in] qubit_count number of the qubits
 * @param[in] angle rotation angle
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * 複数量子ビット（全て指定）のパウリ演算子による回転演算を用いて状態を更新。
 *
 * 複数量子ビット（全て指定）のパウリ演算子 A による回転演算
 *
 * cos ( angle/2 ) I + i sin ( angle/2 ) A 
 *
 * を用いて状態を更新。Pauli_operator_type_list は全ての量子ビットに対するパウリ演算子のリスト。パウリ演算子はI, X, Y, Zがそれぞれ 0,1,2,3 に対応。
 *
 * 例) ５量子ビットに対する YIZXI の場合は、{0,1,3,0,2}（量子ビットの順番は右から数えている）。
 * 
 * @param[in] Pauli_operator_type_list 長さ qubit_count の {0,1,2,3} のリスト。 0,1,2,3 はそれぞれパウリ演算子 I, X, Y, Z に対応。
 * @param[in] qubit_count 量子ビットの数
 * @param[in] angle 回転角度
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 * 
 */
DllExport void multi_qubit_Pauli_rotation_gate_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, double angle, CTYPE* state, ITYPE dim_);

/**
 * \~english
 * Apply multi-qubit Pauli rotation operator to the quantum state with a partial list.
 *
 * Apply multi-qubit Pauli rotation operator to state with a partial list of the Pauli operators. 
 *
 * @param[in] target_qubit_index_list list of the target qubits
 * @param[in] Pauli_operator_type_list list of {0,1,2,3} of length target_qubit_index_count. {0,1,2,3} corresponds to I, X, Y, Z for the target qubits.
 * @param[in] target_qubit_index_count the number of the target qubits
 * @param[in] angle rotation angle
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * 複数量子ビット（部分的に指定）のパウリ演算子による回転演算を用いて状態を更新。
 *
 * 複数量子ビット（部分的に指定）のパウリ演算子 A による回転演算
 *
 * cos ( angle/2 ) I + i sin ( angle/2 ) A 
 *
 * を用いて状態を更新。パウリ演算子Aは、target_qubit_index_list で指定される一部の量子ビットに対するパウリ演算子のリスト Pauli_operator_type_list によって定義。回転角には(1/2)の因子はついていない。パウリ演算子はI, X, Y, Zがそれぞれ 0,1,2,3 に対応。
 *
 * 例) ５量子ビットに対する YIZXI の場合は、target_qubit_index_list = {1,2,4}, Pauli_operator_type_list = {1,3,2}（量子ビットの順番は右から数えている）。
 * 
 * @param[in] target_qubit_index_list 作用する量子ビットのインデックス
 * @param[in] Pauli_operator_type_list 長さ target_qubit_index_countの {0,1,2,3} のリスト。0,1,2,3は、それぞれパウリ演算子 I, X, Y, Z 対応。
 * @param[in] target_qubit_index_count 作用する量子ビットの数
 * @param[in] angle 回転角
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 * 
 */
DllExport void multi_qubit_Pauli_rotation_gate_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, double angle, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply a two-qubit arbitrary gate.
 *
 * Apply a two-qubit arbitrary gate defined by a dense matrix as a one-dimentional array matrix[].
 * @param[in] target_qubit_index1 index of the first target qubit
 * @param[in] target_qubit_index2 index of the second target qubit
 * @param[in] matrix description of a multi-qubit gate as a one-dimensional array with length 16
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * 任意の２量子ビット演算を作用させて状態を更新。
 *
 * 任意の２量子ビット演算を作用させて状態を更新。演算は、その行列成分を１次元配列として matrix[] で与える。(j,k)成分は、matrix[dim*j+k]に対応。
 *
 * @param[in] target_qubit_index1 作用する量子ビットの一つ目の添え字
 * @param[in] target_qubit_index2 作用する量子ビットの二つ目の添え字
 * @param[in] matrix 複数量子ビット演算を定義する長さ 16 の一次元配列。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void two_qubit_dense_matrix_gate(UINT target_qubit_index1, UINT target_qubit_index2, const CTYPE matrix[16], CTYPE *state, ITYPE dim);

/**
 * \~english
 * Apply a multi-qubit arbitrary gate.
 *
 * Apply a multi-qubit arbitrary gate defined by a dense matrix as a one-dimentional array matrix[].
 * @param[in] target_qubit_index_list list of the target qubits
 * @param[in] target_qubit_index_count the number of the target qubits
 * @param[in] matrix description of a multi-qubit gate as a one-dimensional array
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * 任意の複数量子ビット演算を作用させて状態を更新。
 *
 * 任意の複数量子ビット演算を作用させて状態を更新。演算は、その行列成分を１次元配列として matrix[] で与える。(j,k)成分は、matrix[dim*j+k]に対応。
 *
 * 例）パウリX演算子：{0, 1, 1, 0}、アダマール演算子：{1/sqrt(2),1/sqrt(2),1/sqrt(2),-1/sqrt(2)}、
 * 
 * CNOT演算:
 *
 * {1,0,0,0,
 *
 *  0,1,0,0,
 *
 *  0,0,0,1,
 *
 *  0,0,1,0}
 *
 * @param[in] target_qubit_index_list 作用する量子ビットのリスト
 * @param[in] target_qubit_index_count 作用する量子ビットの数
 * @param[in] matrix 複数量子ビット演算を定義する長さ 2^(2* target_qubit_index_count) の一次元配列。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 * 
 */
DllExport void multi_qubit_dense_matrix_gate(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply a single-qubit controlled multi-qubit gate.
 *
 * Apply a single-qubit controlled multi-qubit gate. The multi-qubit gate is by a dense matrix as a one-dimentional array matrix[].
 * @param[in] control_qubit_index index of the control qubit
 * @param[in] control_value value of the control qubit
 * @param[in] target_qubit_index_list list of the target qubits
 * @param[in] target_qubit_index_count the number of the target qubits
 * @param[in] matrix description of a multi-qubit gate as a one-dimensional array
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * １つの制御量子ビットによる任意の複数量子ビット演算の制御演算を作用させて状態を更新。
 *
 *  １つの制御量子ビットによる任意の複数量子ビット演算の制御演算を作用させて状態を更新。制御量子ビットが 0 もしくは 1 のどちらの場合に作用するかは control_value によって指定。複数量子ビット演算は、その行列成分を１次元配列として matrix[] で与える。(j,k)成分は、matrix[dim*j+k]に対応。
 *
 * 例）パウリX演算子：{0, 1, 1, 0}、アダマール演算子：{1/sqrt(2),1/sqrt(2),1/sqrt(2),-1/sqrt(2)}、
 * 
 * CNOT演算:
 *
 * {1,0,0,0,
 *
 *  0,1,0,0,
 *
 *  0,0,0,1,
 *
 *  0,0,1,0}
 *
 * @param[in] control_qubit_index 制御量子ビットのインデックス
 * @param[in] control_value 制御量子ビットの値
 * @param[in] target_qubit_index_list 作用する量子ビットのリスト
 * @param[in] target_qubit_index_count 作用する量子ビットの数
 * @param[in] matrix 複数量子ビット演算を定義する長さ 2^(2* target_qubit_index_count) の一次元配列。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 * 
 */
DllExport void single_qubit_control_multi_qubit_dense_matrix_gate(UINT control_qubit_index, UINT control_value, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply a multi-qubit controlled multi-qubit gate.
 *
 * Apply a multi-qubit controlled multi-qubit gate. The multi-qubit gate is by a dense matrix as a one-dimentional array matrix[].
 *
 * @param[in] control_qubit_index_list list of the indexes of the control qubits
 * @param[in] control_value_list list of the vlues of the control qubits
 * @param[in] target_qubit_index_list list of the target qubits
 * @param[in] target_qubit_index_count the number of the target qubits
 * @param[in] matrix description of a multi-qubit gate as a one-dimensional array
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 * 
 *
 * \~japanese-en
 * 複数の制御量子ビットによる任意の複数量子ビット演算の制御演算を作用させて状態を更新。
 *
 *  複数の制御量子ビットによる任意の複数量子ビット演算の制御演算を作用させて状態を更新。制御量子ビットは、control_qubit_index_listで指定され、制御演算が作用する control_qubit_index_count 個の制御量子ビットの値は、 control_value_list によって指定。複数量子ビット演算は、その行列成分を１次元配列として matrix[] で与える。(j,k)成分は、matrix[dim*j+k]に対応。
 *
 * 例）パウリX演算子：{0, 1, 1, 0}、アダマール演算子：{1/sqrt(2),1/sqrt(2),1/sqrt(2),-1/sqrt(2)}、
 * 
 * CNOT演算:
 *
 * {1,0,0,0,
 *
 *  0,1,0,0,
 *
 *  0,0,0,1,
 *
 *  0,0,1,0}
 *
 * @param[in] control_qubit_index_list 制御量子ビットのインデックスのリスト
 * @param[in] control_value_list 制御演算が作用する制御量子ビットの値のリスト
 * @param[in] control_qubit_index_count 制御量子ビットの数
 * @param[in] target_qubit_index_list ターゲット量子ビットのリスト
 * @param[in] target_qubit_index_count ターゲット量子ビットの数
 * @param[in] matrix 複数量子ビット演算を定義する長さ 2^(2*target_qubit_index_count) の一次元配列。
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 * 
 */
DllExport void multi_qubit_control_multi_qubit_dense_matrix_gate(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim);



/**
 * \~english
 * Reflect state according to another given state.
 *
 * Reflect state according to another given state. When reflect quantum state |a> to state |s>, unitary operator give by 2|s><s|-I is applied to |a>.
 *
 * @param[in] reflection_state quantum state to characterize reflection unitary operator
 * @param[in,out] state quantum state to update
 * @param[in] dim dimension
 *
 *
 * \~japanese-en
 * reflection gateを作用する。
 *
 * 与えられた状態にreflection gateを作用する。量子状態|a>を量子状態|s>に関してreflectするとは、量子状態|a>に対して2|s><s|-Iというユニタリ操作をすることに対応する。
 *
 * @param[in] reflection_state reflection gateのユニタリを特徴づける量子状態
 * @param[in,out] state quantum 更新する量子状態
 * @param[in] dim 次元
 *
 */
DllExport void reflection_gate(const CTYPE* reflection_state, CTYPE* state, ITYPE dim);




////////////////////////////////

///QFT
/**
 * update quantum state with one-qubit controlled dense matrix gate
 * 
 * update quantum state with one-qubit controlled dense matrix gate
 * @param[in] angle rotation angle
 * @param[in] c_bit index of control qubit
 * @param[in] t_bit index of target qubit
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 */
DllExport void CUz_gate(double angle, UINT c_bit, UINT t_bit, CTYPE *state, ITYPE dim);
/**
 * update quantum state with large dense matrix
 * 
 * update quantum state with large dense matrix
 * @param[in] k ???
 * @param[in] Nbits the number of qubits
 * @param[in] doSWAP ???
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 */
DllExport void qft(UINT k, UINT Nbits, int doSWAP, CTYPE *state, ITYPE dim);
/**
 * update quantum state with large dense matrix
 * 
 * update quantum state with large dense matrix
 * @param[in] k ???
 * @param[in] Nbits the number of qubits
 * @param[in] doSWAP ???
 * @param[in,out] state quantum state
 * @param[in] dim dimension
 */
DllExport void inverse_qft(UINT k, UINT Nbits, int doSWAP, CTYPE *state, ITYPE dim);
