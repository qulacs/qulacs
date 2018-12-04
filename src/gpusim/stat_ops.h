#ifndef _STAT_OPS_CU_H_
#define _STAT_OPS_CU_H_

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
//#include <cuda.h>

#include <cuComplex.h>
#include "util_common.h"

extern "C" DllExport __host__ double state_norm_host(void *state, ITYPE dim);
extern "C" DllExport CTYPE inner_product_host(void *psi, void *phi, ITYPE dim);
extern "C" DllExport __host__ double expectation_value_single_qubit_Pauli_operator_host(unsigned int operator_index, unsigned int targetQubitIndex, GTYPE *psi_gpu, ITYPE dim);
extern "C" DllExport __host__ void multi_Z_gate_host(int* gates, GTYPE *psi_gpu, ITYPE dim, int n_qubits);
extern "C" DllExport __host__ double multipauli_get_expectation_value_host(unsigned int* gates, GTYPE *psi_gpu, ITYPE dim, int n_qubits);
extern "C" DllExport __host__ double M0_prob_host(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ double M1_prob_host(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ double marginal_prob_host(UINT* sorted_target_qubit_index_list, UINT* measured_value_list, UINT target_qubit_index_count, void* state, ITYPE dim);
extern "C" DllExport __host__ double expectation_value_multi_qubit_Pauli_operator_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,UINT pivot_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ double expectation_value_multi_qubit_Pauli_operator_Z_mask_host(ITYPE phase_flip_mask, void* state, ITYPE dim);
extern "C" DllExport __host__ double expectation_value_multi_qubit_Pauli_operator_partial_list_host(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, void* state, ITYPE dim);
extern "C" DllExport __host__ double expectation_value_multi_qubit_Pauli_operator_whole_list_host(const UINT* Pauli_operator_type_list, UINT qubit_count, void* state, ITYPE dim);
extern "C" DllExport __host__ CTYPE transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, void* state_bra, void* state_ket, ITYPE dim);
extern "C" DllExport __host__ CTYPE transition_amplitude_multi_qubit_Pauli_operator_Z_mask_host(ITYPE phase_flip_mask, void* state_bra, void* state_ket, ITYPE dim);
extern "C" DllExport __host__ CTYPE transition_amplitude_multi_qubit_Pauli_operator_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, void* state_bra, void* state_ket, ITYPE dim);
extern "C" DllExport __host__ CTYPE transition_amplitude_multi_qubit_Pauli_operator_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, void* state_bra, void* state_ket, ITYPE dim);

#endif // _STAT_OPS_CU_H_
