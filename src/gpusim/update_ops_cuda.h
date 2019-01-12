#ifndef _UPDATE_OPS_CU_H_
#define _UPDATE_OPS_CU_H_

#include "util_export.h"
#include "util_type.h"
#include "stat_ops.h"

//////TODO//////
// multi_qubit_control_single_qubit_dense_matrix_gate_host


// update_ops_named.cu
DllExport void H_gate_host(unsigned int target_qubit_index, void *state, ITYPE dim);
DllExport void X_gate_host(unsigned int target_qubit_index, void *state, ITYPE dim);
DllExport void Y_gate_host(unsigned int target_qubit_index, void *state, ITYPE dim);
DllExport void Z_gate_host(unsigned int target_qubit_index, void *state, ITYPE dim);
DllExport void P0_gate_host(UINT target_qubit_index, void *state, ITYPE dim);
DllExport void P1_gate_host(UINT target_qubit_index, void *state, ITYPE dim);
DllExport void normalize_host(double norm, void* state, ITYPE dim);
DllExport void CZ_gate_host(unsigned int control_qubit_index, unsigned int target_qubit_index, void* state, ITYPE dim);
DllExport void CNOT_gate_host(unsigned int control_qubit_index, unsigned int target_qubit_index, void* state, ITYPE dim);
DllExport void SWAP_gate_host(unsigned int control_qubit_index, unsigned int target_qubit_index, void* state, ITYPE dim);

DllExport void RX_gate_host(UINT target_qubit_index, double angle, void* state, ITYPE dim);
DllExport void RY_gate_host(UINT target_qubit_index, double angle, void* state, ITYPE dim);
DllExport void RZ_gate_host(UINT target_qubit_index, double angle, void* state, ITYPE dim);
DllExport void S_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
DllExport void Sdag_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
DllExport void T_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
DllExport void Tdag_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
DllExport void sqrtX_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
DllExport void sqrtXdag_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
DllExport void sqrtY_gate_host(UINT target_qubit_index, void* state, ITYPE dim);
DllExport void sqrtYdag_gate_host(UINT target_qubit_index, void* state, ITYPE dim);

// update_ops_single
DllExport void single_qubit_Pauli_gate_host(UINT target_qubit_index, UINT Pauli_operator_type, void* state, ITYPE dim);
DllExport void single_qubit_Pauli_rotation_gate_host(unsigned int target_qubit_index, unsigned int op_idx, double angle, void *state, ITYPE dim);
DllExport void single_qubit_dense_matrix_gate_host(unsigned int target_qubit_index, const CPPCTYPE matrix[4], void* state, ITYPE dim);
DllExport void single_qubit_diagonal_matrix_gate_host(unsigned int target_qubit_index, const CPPCTYPE diagonal_matrix[2], void* state, ITYPE dim);
DllExport void single_qubit_control_single_qubit_dense_matrix_gate_host(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, const CPPCTYPE matrix[4], void* state, ITYPE dim);
DllExport void single_qubit_phase_gate_host(unsigned int target_qubit_index, CPPCTYPE phase, void* state, ITYPE dim);
DllExport void multi_qubit_control_single_qubit_dense_matrix_gate_host(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, UINT target_qubit_index, const CPPCTYPE matrix[4], void *state, ITYPE dim);

/*
__device__ void CZ_gate_device(unsigned int control_qubit_index, unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE DIM);
__device__ void SWAP_gate_device(unsigned int target_qubit_index_0, unsigned int target_qubit_index1, GTYPE *psi_gpu, ITYPE dim);
__device__ void single_qubit_dense_matrix_gate_device(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim);
__device__ void single_qubit_diagonal_matrix_gate_device(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim);
__device__ void single_qubit_control_single_qubit_dense_matrix_gate_device(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, GTYPE *state, ITYPE dim);
__device__ void single_qubit_phase_gate_device(unsigned int target_qubit_index, GTYPE phase, GTYPE *state_gpu, ITYPE dim);
*/

//DllExport 


// multi qubit
DllExport void penta_qubit_dense_matrix_gate_host(unsigned int target_qubit_index[5], const CPPCTYPE matrix[1024], void* state, ITYPE dim);
DllExport void quad_qubit_dense_matrix_gate_host(unsigned int target_qubit_index[4], const CPPCTYPE matrix[256], void* state, ITYPE dim);
DllExport void triple_qubit_dense_matrix_gate_host(unsigned int target1_qubit_index, unsigned int target2_qubit_index, unsigned int target3_qubit_index, const CPPCTYPE matrix[64], void* state, ITYPE dim);
DllExport void double_qubit_dense_matrix_gate_host(unsigned int target0_qubit_index, unsigned int target1_qubit_index, const CPPCTYPE matrix[16], void* state, ITYPE dim);
DllExport void multi_qubit_Pauli_gate_Z_mask_host(ITYPE phase_flip_mask, void* state, ITYPE dim);
DllExport void multi_qubit_Pauli_gate_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, void* state, ITYPE dim);
DllExport void multi_qubit_Pauli_rotation_gate_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, void* state, ITYPE dim);
DllExport void multi_qubit_Pauli_rotation_gate_Z_mask_host(ITYPE phase_flip_mask, double angle, void* state, ITYPE dim);
DllExport void multi_qubit_Pauli_gate_partial_list_host(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, void* state, ITYPE dim);
DllExport void multi_qubit_Pauli_gate_whole_list_host(const UINT* Pauli_operator_type_list, UINT qubit_count, void* state, ITYPE dim);
DllExport void multi_qubit_Pauli_rotation_gate_partial_list_host(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, double angle, void* state, ITYPE dim);
DllExport void multi_qubit_Pauli_rotation_gate_whole_list_host(const UINT* Pauli_operator_type_list, UINT qubit_count, double angle, void* state, ITYPE dim);
DllExport void multi_qubit_dense_matrix_gate_host(UINT* target_qubit_index_list, UINT target_qubit_index_count, const CPPCTYPE* matrix, void* state, ITYPE dim);
DllExport void single_qubit_control_multi_qubit_dense_matrix_gate_host(UINT control_qubit_index, UINT control_value, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CPPCTYPE* matrix, void* state, ITYPE dim);
DllExport void multi_qubit_control_multi_qubit_dense_matrix_gate_host(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CPPCTYPE* matrix, void* state, ITYPE dim);

//extern "C" DllExport 

// QCsim.cu
/*
extern "C" DllExport __host__ cudaError U1_gate_host(double lambda, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE DIM);
extern "C" DllExport __host__ cudaError U2_gate_host(double lambda, double phi, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE DIM);
extern "C" DllExport __host__ cudaError U3_gate_host(double lambda, double phi, double theta, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE DIM);
*/

#endif // #ifndef _UPDATE_OPS_CU_H_
