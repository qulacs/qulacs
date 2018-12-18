#ifndef _UTIL_CU_H_
#define _UTIL_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
//#include <cuda_runtime.h>
//#include <cuda.h>

#include <complex>
#include "util_common.h"
//#include "util.cuh"

extern "C" DllExport void get_quantum_state_host(void* state_gpu, void* psi_cpu_copy, ITYPE dim);
extern "C" DllExport void* allocate_quantum_state_host(ITYPE dim);
extern "C" DllExport void initialize_quantum_state_host(void* state_gpu, ITYPE dim);
extern "C" DllExport void release_quantum_state_host(void* state_gpu);
extern "C" DllExport void print_quantum_state_host(void* state, ITYPE dim);
extern "C" DllExport void copy_quantum_state_host(void* state_gpu_copy, void* state_gpu, ITYPE dim);
extern "C" DllExport __host__ void set_computational_basis_host(ITYPE comp_basis, void* state, ITYPE dim);

void get_Pauli_masks_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, 
    ITYPE* bit_flip_mask, ITYPE* phase_flip_mask, UINT* global_phase_90rot_count, UINT* pivot_qubit_index);
void get_Pauli_masks_whole_list(const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, 
    ITYPE* bit_flip_mask, ITYPE* phase_flip_mask, UINT* global_phase_90rot_count, UINT* pivot_qubit_index);


// int cublass_zgemm_wrapper(ITYPE n, CTYPE alpha, const CTYPE *h_A, const CTYPE *h_B, CTYPE beta, CTYPE *h_C);
// int cublas_zgemv_wrapper(ITYPE n, CTYPE alpha, const CTYPE *h_A, const CTYPE *h_x, CTYPE beta, CTYPE *h_y);
// int cublas_zgemv_wrapper(ITYPE n, const CTYPE *h_matrix, GTYPE *d_state);

ITYPE create_control_mask(const UINT* qubit_index_list, const UINT* value_list, UINT size);
UINT* create_sorted_ui_list(const UINT* array, size_t size);
ITYPE* create_matrix_mask_list(const UINT* qubit_index_list, UINT qubit_index_count);
UINT* create_sorted_ui_list_value(const UINT* array, size_t size, UINT value);
UINT* create_sorted_ui_list_list(const UINT* array1, size_t size1, const UINT* array2, size_t size2);
ITYPE insert_zero_to_basis_index(ITYPE basis_index, unsigned int qubit_index);


#endif // #ifndef _QCUDASIM_UTIL_H_
