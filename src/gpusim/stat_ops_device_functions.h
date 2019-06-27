#ifndef _STAT_OPS_CU_DEVICE_H_
#define _STAT_OPS_CU_DEVICE_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include "util_type.h"
#include "util_export.h"
#include "util_type_internal.h"

__global__ void state_norm_gpu(double* ret, GTYPE *state, ITYPE dim);
__global__ void measurement_distribution_entropy_gpu(double* ret, const GTYPE *state, ITYPE dim);
__global__ void state_add_gpu(const GTYPE *state_added, GTYPE *state, ITYPE dim);
__global__ void state_multiply_gpu(const GTYPE coef, GTYPE *state, ITYPE dim);
__global__ void inner_product_gpu(GTYPE *ret, const GTYPE *psi, const GTYPE *phi, ITYPE dim);
__global__ void expectation_value_PauliI_gpu(double *ret, GTYPE *state, unsigned int target_qubit_index, ITYPE dim);
__global__ void expectation_value_PauliX_gpu(double *ret, GTYPE *state, unsigned int target_qubit_index, ITYPE dim);
__global__ void expectation_value_PauliY_gpu(double *ret, GTYPE *state, unsigned int target_qubit_index, ITYPE dim);
__global__ void expectation_value_PauliZ_gpu(double *ret, GTYPE *state, unsigned int target_qubit_index, ITYPE dim);
__global__ void multi_Z_gate_gpu(ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu);
__global__ void multi_Z_get_expectation_value_gpu(GTYPE *ret, ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu);
__global__ void multipauli_get_expectation_value_gpu(GTYPE* ret, ITYPE DIM, GTYPE *psi_gpu, int n_qubits);
__global__ void M0_prob_gpu(double* ret, UINT target_qubit_index, const GTYPE* state, ITYPE dim);
__global__ void M1_prob_gpu(double* ret, UINT target_qubit_index, const GTYPE* state, ITYPE dim);
__global__ void marginal_prob_gpu(double* ret_gpu, const UINT* sorted_target_qubit_index_list, const UINT* measured_value_list, UINT target_qubit_index_count, const GTYPE* state, ITYPE dim);
__global__ void expectation_value_multi_qubit_Pauli_operator_XZ_mask_gpu(double* ret_gpu, ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, GTYPE* state, ITYPE dim);
__global__ void expectation_value_multi_qubit_Pauli_operator_Z_mask_gpu(double* ret_gpu, ITYPE phase_flip_mask, const GTYPE* state, ITYPE dim);
__global__ void transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_gpu(GTYPE* ret_gpu, ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, const GTYPE* state_bra, const GTYPE* state_ket, ITYPE dim);
__global__ void transition_amplitude_multi_qubit_Pauli_operator_Z_mask_gpu(GTYPE* ret, ITYPE phase_flip_mask, GTYPE* state_bra, GTYPE* state_ket, ITYPE dim);


#endif // _STAT_OPS_CU_DEVICE_H_
