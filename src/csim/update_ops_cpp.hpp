#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <functional>
#ifndef _MSC_VER
extern "C"{
#include "type.h"
}
#else
#include "type.h"
#endif

DllExport void double_qubit_dense_matrix_gate(UINT target_qubit_index0, UINT target_qubit_index1, const CTYPE matrix[16], CTYPE* state, ITYPE dim);
DllExport void double_qubit_dense_matrix_gate(UINT target_qubit_index0, UINT target_qubit_index1, const Eigen::Matrix4cd& eigen_matrix, CTYPE* state, ITYPE dim);
DllExport void double_qubit_dense_matrix_gate_eigen(UINT target_qubit_index0, UINT target_qubit_index1, const Eigen::Matrix4cd& eigen_matrix, CTYPE* state, ITYPE dim);

DllExport void multi_qubit_dense_matrix_gate_eigen(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim);
DllExport void multi_qubit_dense_matrix_gate_eigen(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const Eigen::MatrixXcd& eigen_matrix, CTYPE* state, ITYPE dim);
DllExport void multi_qubit_dense_matrix_gate_eigen(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>& eigen_matrix, CTYPE* state, ITYPE dim);

DllExport void multi_qubit_sparse_matrix_gate_eigen(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const Eigen::SparseMatrix<std::complex<double>>& eigen_matrix, CTYPE* state, ITYPE dim);

/**
 * \~english
 * Apply reversible boolean function as a unitary gate.
 *
 * Apply reversible boolean function as a unitary gate. Boolean function is given as a pointer of int -> int function.
 *
 * @param[in] target_qubit_index_list ターゲット量子ビットのリスト
 * @param[in] target_qubit_index_count ターゲット量子ビットの数
 * @param[in] matrix 添え字および対象ビットの次元を受け取ると添え字を返す関数
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 *
 * \~japanese-en
 * 可逆回路関数をユニタリゲートとして作用する
 *
 *  可逆回路関数をユニタリゲートとして作用する。可逆回路関数は添え字を与えると結果の添え字を返す関数。
 *
 * @param[in] target_qubit_index_list ターゲット量子ビットのリスト
 * @param[in] target_qubit_index_count ターゲット量子ビットの数
 * @param[in] matrix 添え字および対象ビットの次元を受け取ると添え字を返す関数
 * @param[in,out] state 量子状態
 * @param[in] dim 次元
 *
 */
DllExport void reversible_boolean_gate(const UINT* target_qubit_index_list, UINT target_qubit_index_count, std::function<ITYPE(ITYPE,ITYPE)> function_ptr, CTYPE* state, ITYPE dim);

