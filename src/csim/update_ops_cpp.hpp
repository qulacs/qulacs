#pragma once

#include <Eigen/Core>
#ifndef _MSC_VER
extern "C"{
#include "type.h"
}
#else
#include "type.h"
#endif

DllExport void multi_qubit_dense_matrix_gate_eigen(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim);
DllExport void multi_qubit_dense_matrix_gate_eigen(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const Eigen::MatrixXcd& eigen_matrix, CTYPE* state, ITYPE dim);
DllExport void multi_qubit_dense_matrix_gate_eigen(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>& eigen_matrix, CTYPE* state, ITYPE dim);
