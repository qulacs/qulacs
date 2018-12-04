#ifndef _UPDATE_OPS_CU_H_
#define _UPDATE_OPS_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
//#include "util.h"
#include "util_common.h"
#include "stat_ops.h"

// update_ops_named.cu
extern "C" DllExport __host__ void H_gate_host(unsigned int target_qubit_index, void *state, ITYPE dim);
extern "C" DllExport __host__ void X_gate_host(unsigned int target_qubit_index, void *state, ITYPE dim);
extern "C" DllExport __host__ void Y_gate_host(unsigned int target_qubit_index, void *state, ITYPE dim);
extern "C" DllExport __host__ void Z_gate_host(unsigned int target_qubit_index, void *state, ITYPE dim);
extern "C" DllExport __host__ void P0_gate_host(UINT target_qubit_index, void *state, ITYPE dim);
extern "C" DllExport __host__ void P1_gate_host(UINT target_qubit_index, void *state, ITYPE dim);
extern "C" DllExport __host__ void normalize_host(double norm, void* state, ITYPE dim);
__device__ void CZ_gate_device(unsigned int control_qubit_index, unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE DIM);
extern "C" DllExport __host__ void CZ_gate_host(unsigned int control_qubit_index, unsigned int target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ void CNOT_gate_host(unsigned int control_qubit_index, unsigned int target_qubit_index, void* state, ITYPE dim);
__device__ void SWAP_gate_device(unsigned int target_qubit_index_0, unsigned int target_qubit_index1, GTYPE *psi_gpu, ITYPE dim);
extern "C" DllExport __host__ void SWAP_gate_host(unsigned int control_qubit_index, unsigned int target_qubit_index, void* state, ITYPE dim);

extern "C" DllExport __host__ void RX_gate_host(UINT target_qubit_index, double angle, void* state, ITYPE dim);
extern "C" DllExport __host__ void RY_gate_host(UINT target_qubit_index, double angle, void* state, ITYPE dim);
extern "C" DllExport __host__ void RZ_gate_host(UINT target_qubit_index, double angle, void* state, ITYPE dim);
extern "C" DllExport __host__ void S_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ void Sdag_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ void T_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ void Tdag_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ void sqrtX_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ void sqrtXdag_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ void sqrtY_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ void sqrtYdag_gate_host(UINT target_qubit_index, void* state, ITYPE dim);

// update_ops_single
extern "C" DllExport __host__ void single_qubit_Pauli_gate_host(UINT target_qubit_index, UINT Pauli_operator_type, void* state, ITYPE dim);
extern "C" DllExport __host__ void single_qubit_Pauli_rotation_gate_host(unsigned int target_qubit_index, unsigned int op_idx, double angle, void *state, ITYPE dim);
__device__ void single_qubit_dense_matrix_gate_device(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim);
extern "C" DllExport __host__ void single_qubit_dense_matrix_gate_host(unsigned int target_qubit_index, CTYPE matrix[4], void* state, ITYPE dim);
__device__ void single_qubit_diagonal_matrix_gate_device(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim);
extern "C" DllExport __host__ void single_qubit_diagonal_matrix_gate_host(unsigned int target_qubit_index, const CTYPE diagonal_matrix[2], void* state, ITYPE dim);
__device__ void single_qubit_control_single_qubit_dense_matrix_gate_device(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, GTYPE *state, ITYPE dim);
extern "C" DllExport __host__ void single_qubit_control_single_qubit_dense_matrix_gate_host(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, CTYPE matrix[4], void* state, ITYPE dim);
__device__ void single_qubit_phase_gate_device(unsigned int target_qubit_index, GTYPE phase, GTYPE *state_gpu, ITYPE dim);
extern "C" DllExport __host__ void single_qubit_phase_gate_host(unsigned int target_qubit_index, CTYPE phase, GTYPE *state, ITYPE dim);

//extern "C" DllExport 


// multi qubit
extern "C" DllExport __host__ void penta_qubit_dense_matrix_gate_host(unsigned int target_qubit_index[5], CTYPE matrix[1024], void* state, ITYPE dim);
extern "C" DllExport  __host__ void quad_qubit_dense_matrix_gate_host(unsigned int target_qubit_index[4], CTYPE matrix[256], void* state, ITYPE dim);
extern "C" DllExport __host__ void triple_qubit_dense_matrix_gate_host(unsigned int target1_qubit_index, unsigned int target2_qubit_index, unsigned int target3_qubit_index, CTYPE matrix[64], void* state, ITYPE dim);
extern "C" DllExport __host__ void double_qubit_dense_matrix_gate_host(unsigned int target1_qubit_index, unsigned int target0_qubit_index, CTYPE matrix[16], void* state, ITYPE dim);
extern "C" DllExport __host__ void multi_qubit_Pauli_gate_Z_mask_host(ITYPE phase_flip_mask, void* state, ITYPE dim);
extern "C" DllExport __host__ void multi_qubit_Pauli_gate_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ void multi_qubit_Pauli_rotation_gate_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, void* state, ITYPE dim);
extern "C" DllExport __host__ void multi_qubit_Pauli_rotation_gate_Z_mask_host(ITYPE phase_flip_mask, double angle, void* state, ITYPE dim);
extern "C" DllExport __host__ void multi_qubit_Pauli_gate_partial_list_host(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, void* state, ITYPE dim);
extern "C" DllExport __host__ void multi_qubit_Pauli_gate_whole_list_host(const UINT* Pauli_operator_type_list, UINT qubit_count, void* state, ITYPE dim);
extern "C" DllExport __host__ void multi_qubit_Pauli_rotation_gate_partial_list_host(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, double angle, void* state, ITYPE dim);
extern "C" DllExport __host__ void multi_qubit_Pauli_rotation_gate_whole_list_host(const UINT* Pauli_operator_type_list, UINT qubit_count, double angle, void* state, ITYPE dim);
extern "C" DllExport __host__ void multi_qubit_dense_matrix_gate(UINT* target_qubit_index_list, UINT target_qubit_index_count, CTYPE* matrix, void* state, ITYPE dim);
extern "C" DllExport __host__ void single_qubit_control_multi_qubit_dense_matrix_gate_host(UINT control_qubit_index, UINT control_value, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, void* state, ITYPE dim);
extern "C" DllExport __host__ void multi_qubit_control_multi_qubit_dense_matrix_gate_host(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CTYPE* matrix, void* state, ITYPE dim);

//extern "C" DllExport 

// QCsim.cu
/*
extern "C" DllExport __host__ cudaError U1_gate_host(double lambda, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE DIM);
extern "C" DllExport __host__ cudaError U2_gate_host(double lambda, double phi, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE DIM);
extern "C" DllExport __host__ cudaError U3_gate_host(double lambda, double phi, double theta, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE DIM);
*/

#endif // #ifndef _UPDATE_OPS_CU_H_
