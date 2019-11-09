#ifndef _UTIL_FUNC_H_
#define _UTIL_FUNC_H_

#include <complex>
#include "util_export.h"
#include "util_type.h"

DllExport int get_num_device();
DllExport void set_device(unsigned int device_num);
DllExport int get_current_device();
DllExport void set_computational_basis_host(ITYPE comp_basis, void* state, ITYPE dim, void* stream, unsigned int device_number);
DllExport void copy_quantum_state_from_device_to_device(void* state_gpu_copy, const void* state_gpu, ITYPE dim, void* stream, unsigned int device_number);
DllExport void copy_quantum_state_from_host_to_device(void* state_gpu_copy, const void* state, ITYPE dim, void* stream, unsigned int device_number);
DllExport void copy_quantum_state_from_cppstate_host(void* state_gpu_copy, const CPPCTYPE* cppstate, ITYPE dim, void* stream, UINT device_number);
DllExport void copy_quantum_state_from_device_to_host(void* state_cpu_copy, const void* state_gpu_original, ITYPE dim, void* stream, unsigned int device_number);
DllExport void get_quantum_state_host(void* state_gpu, void* psi_cpu_copy, ITYPE dim, void* stream, unsigned int device_number);
DllExport void print_quantum_state_host(void* state, ITYPE dim, unsigned int device_number);

ITYPE insert_zero_to_basis_index_gsim(ITYPE basis_index, unsigned int qubit_index);
void get_Pauli_masks_partial_list_gsim(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count,
    ITYPE* bit_flip_mask, ITYPE* phase_flip_mask, UINT* global_phase_90rot_count, UINT* pivot_qubit_index);
void get_Pauli_masks_whole_list_gsim(const UINT* Pauli_operator_type_list, UINT target_qubit_index_count,
    ITYPE* bit_flip_mask, ITYPE* phase_flip_mask, UINT* global_phase_90rot_count, UINT* pivot_qubit_index);
ITYPE* create_matrix_mask_list_gsim(const UINT* qubit_index_list, UINT qubit_index_count);
ITYPE create_control_mask_gsim(const UINT* qubit_index_list, const UINT* value_list, UINT size);
UINT* create_sorted_ui_list_gsim(const UINT* array, size_t size);
UINT* create_sorted_ui_list_value_gsim(const UINT* array, size_t size, UINT value);
UINT* create_sorted_ui_list_list_gsim(const UINT* array1, size_t size1, const UINT* array2, size_t size2);
// int cublas_zgemm_wrapper(ITYPE n, CPPCTYPE alpha, const CPPCTYPE *h_A, const CPPCTYPE *h_B, CPPCTYPE beta, CPPCTYPE *h_C);
// int cublas_zgemv_wrapper(ITYPE n, CPPCTYPE alpha, const CPPCTYPE *h_A, const CPPCTYPE *h_x, CPPCTYPE beta, CPPCTYPE *h_y);
// int cublas_zgemv_wrapper(ITYPE n, const CPPCTYPE *h_matrix, GTYPE *d_state);
// int cublas_zgemv_wrapper(ITYPE n, const GTYPE *d_matrix, GTYPE *d_state);

#endif // #ifndef _QCUDASIM_UTIL_H_
