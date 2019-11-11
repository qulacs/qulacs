#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
//#include "util.h"
#include "util.cuh"
#include "util_type.h"
#include "util_type_internal.h"
#include "util_func.h"
#include "update_ops_cuda.h"
#include "update_ops_cuda_device_functions.h"
#include <assert.h>

__constant__ GTYPE matrix_const_gpu[4];
__constant__ UINT insert_index_list_gpu[30]; 

__host__ void single_qubit_Pauli_gate_host(UINT target_qubit_index, UINT Pauli_operator_type, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	switch (Pauli_operator_type) {
	case 0:
		break;
	case 1:
		X_gate_host(target_qubit_index, state, dim, stream, device_number);
		break;
	case 2:
		Y_gate_host(target_qubit_index, state, dim, stream, device_number);
		break;
	case 3:
		Z_gate_host(target_qubit_index, state, dim, stream, device_number);
		break;
	default:
		fprintf(stderr, "invalid Pauli operation is called");
		assert(0);
	}
}

__host__ void single_qubit_Pauli_rotation_gate_host(unsigned int target_qubit_index, unsigned int op_idx, double angle, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	CPPCTYPE PAULI_MATRIX[4][4] = {
		{ CPPCTYPE(1, 0), CPPCTYPE(0, 0), CPPCTYPE(0, 0), CPPCTYPE(1, 0) },
		{ CPPCTYPE(0, 0), CPPCTYPE(1, 0), CPPCTYPE(1, 0), CPPCTYPE(0, 0) },
		{ CPPCTYPE(0, 0), CPPCTYPE(0, -1), CPPCTYPE(0, 1), CPPCTYPE(0, 0) },
		{ CPPCTYPE(1, 0), CPPCTYPE(0, 0), CPPCTYPE(0, 0), CPPCTYPE(-1, 0) }
	};
	CPPCTYPE rotation_gate[4];

	rotation_gate[0] = CPPCTYPE(
		cos(angle / 2) - sin(angle / 2)* PAULI_MATRIX[op_idx][0].imag(),
		sin(angle / 2) * PAULI_MATRIX[op_idx][0].real()
	);
	rotation_gate[1] = CPPCTYPE(
		-sin(angle / 2)* PAULI_MATRIX[op_idx][1].imag(),
		sin(angle / 2) * PAULI_MATRIX[op_idx][1].real()
	);
	rotation_gate[2] = CPPCTYPE(
		-sin(angle / 2)* PAULI_MATRIX[op_idx][2].imag(),
		sin(angle / 2) * PAULI_MATRIX[op_idx][2].real()
	);
	rotation_gate[3] = CPPCTYPE(
		cos(angle / 2) - sin(angle / 2)* PAULI_MATRIX[op_idx][3].imag(),
		sin(angle / 2) * PAULI_MATRIX[op_idx][3].real()
	);

	single_qubit_dense_matrix_gate_host(target_qubit_index, rotation_gate, state_gpu, dim, stream, device_number);
	state = reinterpret_cast<void*>(state_gpu);
}

__device__ void single_qubit_dense_matrix_gate_device(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim){
	ITYPE basis0, basis1;
	ITYPE half_dim = dim >> 1;
	GTYPE tmp;
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < half_dim){
		basis0 = (j >> target_qubit_index);
		basis0 = basis0 << (target_qubit_index + 1);
		basis0 += j & ((1ULL << target_qubit_index) - 1);
		basis1 = basis0 ^ (1ULL << target_qubit_index);

		tmp = state_gpu[basis0];
		state_gpu[basis0] = cuCadd(cuCmul(matrix_const_gpu[0], tmp), cuCmul(matrix_const_gpu[1], state_gpu[basis1]));
		state_gpu[basis1] = cuCadd(cuCmul(matrix_const_gpu[2], tmp), cuCmul(matrix_const_gpu[3], state_gpu[basis1]));
	}
}

__global__ void single_qubit_dense_matrix_gate_gpu(GTYPE mat0, GTYPE mat1, GTYPE mat2, GTYPE mat3, unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim){
	ITYPE basis0, basis1;
	ITYPE half_dim = dim >> 1;
	GTYPE tmp0, tmp1;
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < half_dim){
		basis0 = (j >> target_qubit_index);
		basis0 = basis0 << (target_qubit_index + 1);
		basis0 += j & ((1ULL << target_qubit_index) - 1);
		basis1 = basis0 ^ (1ULL << target_qubit_index);

		tmp0 = state_gpu[basis0];
		tmp1 = state_gpu[basis1];
		state_gpu[basis0] = cuCadd(cuCmul(mat0, tmp0), cuCmul(mat1, tmp1));
		state_gpu[basis1] = cuCadd(cuCmul(mat2, tmp0), cuCmul(mat3, tmp1));
	}
}

__host__ void single_qubit_dense_matrix_gate_host(unsigned int target_qubit_index, const CPPCTYPE matrix[4], void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	cudaError cudaStatus;

	ITYPE loop_dim = dim >> 1;
	unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;
	GTYPE mat[4];
	for (int i = 0; i < 4; ++i) mat[i] = make_cuDoubleComplex(matrix[i].real(), matrix[i].imag());
	single_qubit_dense_matrix_gate_gpu << < grid, block, 0, *cuda_stream >> > (mat[0], mat[1], mat[2], mat[3], target_qubit_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__device__ void single_qubit_diagonal_matrix_gate_device(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim) {
    ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(state_index<dim){
		state_gpu[state_index] = cuCmul(matrix_const_gpu[(state_index >> target_qubit_index) & 1], state_gpu[state_index]);
	}
}

__global__ void single_qubit_diagonal_matrix_gate_gpu(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim) {
	single_qubit_diagonal_matrix_gate_device(target_qubit_index, state_gpu, dim);
}

__host__ void single_qubit_diagonal_matrix_gate_host(unsigned int target_qubit_index, const CPPCTYPE diagonal_matrix[2], void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	cudaError cudaStatus;
	checkCudaErrors(cudaMemcpyToSymbolAsync(matrix_const_gpu, diagonal_matrix, sizeof(GTYPE) * 2, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);

	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;

	single_qubit_diagonal_matrix_gate_gpu << <grid, block, 0, *cuda_stream >> > (target_qubit_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__device__ void single_qubit_control_single_qubit_dense_matrix_gate_device(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, GTYPE *state, ITYPE dim) {
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
    const ITYPE loop_dim = dim>>2;
    // target mask
    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = (1ULL << control_qubit_index) * control_value;
    // insert index
    const unsigned int min_qubit_index = (control_qubit_index<target_qubit_index) ? control_qubit_index : target_qubit_index;
    const unsigned int max_qubit_index = (control_qubit_index>target_qubit_index) ? control_qubit_index : target_qubit_index;

	if(state_index<loop_dim){
        // create base index
		ITYPE basis_c_t0 = state_index;

        basis_c_t0 = insert_zero_to_basis_index_device(basis_c_t0, min_qubit_index);
        basis_c_t0 = insert_zero_to_basis_index_device(basis_c_t0, max_qubit_index);
        // flip control
        basis_c_t0 ^= control_mask;
        // gather index
        ITYPE basis_c_t1 = basis_c_t0 ^ target_mask;
        // fetch values
        GTYPE cval_c_t0 = state[basis_c_t0];
        GTYPE cval_c_t1 = state[basis_c_t1];
        // set values
        state[basis_c_t0] = cuCadd(cuCmul(matrix_const_gpu[0], cval_c_t0), cuCmul(matrix_const_gpu[1], cval_c_t1));
        state[basis_c_t1] = cuCadd(cuCmul(matrix_const_gpu[2], cval_c_t0), cuCmul(matrix_const_gpu[3], cval_c_t1));
    }
}

__global__ void single_qubit_control_single_qubit_dense_matrix_gate_gpu(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim) {
	single_qubit_control_single_qubit_dense_matrix_gate_device(control_qubit_index, control_value, target_qubit_index, state_gpu, dim);
}

__host__ void single_qubit_control_single_qubit_dense_matrix_gate_host(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, const CPPCTYPE matrix[4], void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);

	cudaError cudaStatus;
	checkCudaErrors(cudaMemcpyToSymbolAsync(matrix_const_gpu, matrix, sizeof(GTYPE) * 4, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);

	ITYPE quad_dim = dim >> 2;
	unsigned int block = quad_dim <= 1024 ? quad_dim : 1024;
	unsigned int grid = quad_dim / block;

	single_qubit_control_single_qubit_dense_matrix_gate_gpu << <grid, block, 0, *cuda_stream >> > (control_qubit_index, control_value, target_qubit_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__device__ void single_qubit_phase_gate_device(unsigned int target_qubit_index, GTYPE phase, GTYPE *state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	// target tmask
	const ITYPE mask = 1ULL << target_qubit_index;
	
	// loop varaibles
	const ITYPE loop_dim = dim>>1;
	
	if(state_index<loop_dim){
		// create index
		ITYPE basis_1 = insert_zero_to_basis_index_device(state_index, target_qubit_index) ^ mask;
	
		// set values
		state_gpu[basis_1] = cuCmul(state_gpu[basis_1], phase);
	}
}

__global__ void single_qubit_phase_gate_gpu(unsigned int target_qubit_index, GTYPE phase, GTYPE *state_gpu, ITYPE dim){
	single_qubit_phase_gate_device(target_qubit_index, phase, state_gpu, dim);
}

__host__ void single_qubit_phase_gate_host(unsigned int target_qubit_index, CPPCTYPE phase, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	GTYPE phase_gtype;
	cudaError cudaStatus;

	phase_gtype = make_cuDoubleComplex(phase.real(), phase.imag());
	ITYPE half_dim = dim >> 1;
	unsigned int block = half_dim <= 1024 ? half_dim : 1024;
	unsigned int grid = half_dim / block;

	single_qubit_phase_gate_gpu << <grid, block, 0, *cuda_stream >> > (target_qubit_index, phase_gtype, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__global__ void multi_qubit_control_single_qubit_dense_matrix_gate(const ITYPE control_mask, UINT control_qubit_index_count, UINT target_qubit_index, GTYPE *state, ITYPE dim) {
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;

    // insert index list
    const UINT insert_index_list_count = control_qubit_index_count + 1;

    // target mask
    const ITYPE target_mask = 1ULL << target_qubit_index;

    const ITYPE loop_dim = dim >> insert_index_list_count;

    if(state_index<loop_dim){

        // create base index
        ITYPE basis_c_t0 = state_index;
        for(UINT cursor = 0 ; cursor < insert_index_list_count ; ++cursor){
            basis_c_t0 = insert_zero_to_basis_index_device(basis_c_t0, insert_index_list_gpu[cursor]);
        }

        // flip controls
        basis_c_t0 ^= control_mask;

        // gather target
        ITYPE basis_c_t1 = basis_c_t0 ^ target_mask;

        // fetch values
        GTYPE cval_c_t0 = state[basis_c_t0];
        GTYPE cval_c_t1 = state[basis_c_t1];

        // set values
        state[basis_c_t0] = cuCadd( cuCmul( matrix_const_gpu[0], cval_c_t0), cuCmul( matrix_const_gpu[1], cval_c_t1));
        state[basis_c_t1] = cuCadd( cuCmul( matrix_const_gpu[2], cval_c_t0), cuCmul( matrix_const_gpu[3], cval_c_t1));
    }
}

__host__ void multi_qubit_control_single_qubit_dense_matrix_gate_host(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, UINT target_qubit_index, const CPPCTYPE matrix[4], void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);

	// insert index list
	const UINT insert_index_list_count = control_qubit_index_count + 1;
	UINT* insert_index_list = create_sorted_ui_list_value_gsim(control_qubit_index_list, control_qubit_index_count, target_qubit_index);

	// control mask
	ITYPE control_mask = create_control_mask_gsim(control_qubit_index_list, control_value_list, control_qubit_index_count);

	// loop varaibles
	const ITYPE loop_dim = dim >> insert_index_list_count;

	unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;

	checkCudaErrors(cudaMemcpyToSymbol(matrix_const_gpu, matrix, sizeof(GTYPE) * 4), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpyToSymbol(insert_index_list_gpu, insert_index_list, sizeof(UINT)*insert_index_list_count), __FILE__, __LINE__);

	multi_qubit_control_single_qubit_dense_matrix_gate << < grid, block, 0, *cuda_stream >> > (control_mask, control_qubit_index_count, target_qubit_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
	free(insert_index_list);
	state = reinterpret_cast<void*>(state_gpu);
}
