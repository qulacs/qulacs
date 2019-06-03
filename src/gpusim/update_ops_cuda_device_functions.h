#ifndef _UPDATE_OPS_CU_DEVICE_H_
#define _UPDATE_OPS_CU_DEVICE_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include "util_export.h"
#include "util_type.h"
#include "util_type_internal.h"

// update_ops_named
__global__ void H_gate_gpu(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void X_gate_gpu(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void Y_gate_gpu(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void Z_gate_gpu(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE DIM);
__global__ void CZ_gate_gpu(unsigned int large_index, unsigned int small_index, GTYPE *state_gpu, ITYPE DIM);
__global__ void CNOT_gate_gpu(unsigned int control_qubit_index, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim);
__global__ void SWAP_gate_gpu(unsigned int target_qubit_index0, unsigned int target_qubit_index1, GTYPE *state_gpu, ITYPE dim);
__global__ void P0_gate_gpu(UINT target_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void P1_gate_gpu(UINT target_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void normalize_gpu(const double normalize_factor, GTYPE *state_gpu, ITYPE dim);


// update_ops_single
__global__ void single_qubit_dense_matrix_gate_shared_gpu(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void single_qubit_dense_matrix_gate_gpu(GTYPE mat0, GTYPE mat1, GTYPE mat2, GTYPE mat3, unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void single_qubit_diagonal_matrix_gate_gpu(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void single_qubit_control_single_qubit_dense_matrix_gate_gpu(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void single_qubit_phase_gate_gpu(unsigned int target_qubit_index, GTYPE phase, GTYPE *state_gpu, ITYPE dim);
__global__ void multi_qubit_control_single_qubit_dense_matrix_gate(const ITYPE control_mask, UINT control_qubit_index_count, UINT target_qubit_index, GTYPE *state, ITYPE dim);

// update_ops_multi
__global__ void penta_qubit_dense_matrix_gate_gpu(GTYPE *state_gpu, ITYPE dim);
__global__ void quad_qubit_dense_matrix_gate_shared_gpu(GTYPE *state_gpu, ITYPE dim);
__global__ void quad_qubit_dense_matrix_gate_gpu(unsigned int target0_qubit_index, unsigned int target1_qubit_index, unsigned int target2_qubit_index, unsigned int target3_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void triple_qubit_dense_matrix_gate_shared_gpu(unsigned int target0_qubit_index, unsigned int target1_qubit_index, unsigned int target2_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void triple_qubit_dense_matrix_gate_gpu(unsigned int target0_qubit_index, unsigned int target1_qubit_index, unsigned int target2_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void double_qubit_dense_matrix_gate_gpu(unsigned int target0_qubit_index, unsigned int target1_qubit_index, GTYPE *state_gpu, ITYPE dim);
__global__ void multi_qubit_Pauli_gate_Z_mask_gpu(ITYPE phase_flip_mask, GTYPE* state_gpu, ITYPE dim);
__global__ void multi_qubit_Pauli_gate_XZ_mask_gpu(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, GTYPE* state_gpu, ITYPE dim);
__global__ void multi_qubit_Pauli_rotation_gate_XZ_mask_gpu(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim);
__global__ void multi_qubit_Pauli_rotation_gate_Z_mask_gpu(ITYPE phase_flip_mask, double angle, GTYPE* state_gpu, ITYPE dim);
__global__ void multi_qubit_dense_matrix_gate_shared_gpu(UINT target_qubit_index_count, GTYPE *state_gpu, ITYPE dim);
__global__ void multi_qubit_dense_matrix_gate_shared_gpu(UINT target_qubit_index_count, GTYPE* matrix_gpu, GTYPE *state_gpu, ITYPE dim);
__global__ void multi_qubit_dense_matrix_gate_half_shared_gpu(UINT target_qubit_index_count, GTYPE* matrix_gpu, GTYPE *state_gpu, ITYPE dim);
__global__ void multi_qubit_dense_matrix_gate_gpu(UINT target_qubit_index_count, GTYPE* matrix_gpu, GTYPE* state_gpu, GTYPE* state_gpu_copy, ITYPE dim);
__global__ void single_qubit_control_multi_qubit_dense_matrix_gate_const_gpu(UINT control_qubit_index, UINT control_value, UINT target_qubit_index_count, GTYPE* state, ITYPE dim);
__global__ void single_qubit_control_multi_qubit_dense_matrix_gate_const_gpu(UINT control_qubit_index, UINT control_value, UINT target_qubit_index_count, const GTYPE* matrix, GTYPE* state, ITYPE dim);
__global__ void multi_qubit_control_multi_qubit_dense_matrix_gate_const_gpu(ITYPE control_mask, UINT target_qubit_index_count, ITYPE control_qubit_index_count, GTYPE* state, ITYPE dim);
__global__ void multi_qubit_control_multi_qubit_dense_matrix_gate_const_gpu(ITYPE control_mask, UINT target_qubit_index_count, ITYPE control_qubit_index_count, const GTYPE* matrix, GTYPE* state, ITYPE dim);



#endif // #ifndef _UPDATE_OPS_CU_DEVICE_H_
