#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include "util_type.h"
#include "util_type_internal.h"
#include "util_func.h"
#include "util.cuh"
#include "update_ops_cuda.h"
#include <cublas_v2.h>
#include <stdio.h>
#include <algorithm>
#include <assert.h>
#include <functional>

// maximum # of GTYPE elements allocating on constant memory: 4096 
__constant__ GTYPE matrix_const_gpu[1024];
__constant__ ITYPE matrix_mask_list_gpu[1024];
__constant__ UINT sorted_insert_index_list_gpu[15];
__constant__ UINT target_index_list_gpu[15];

/**  vqcsim からの移植
 * perform multi_qubit_Pauli_gate with XZ mask.
 * 
 * This function assumes bit_flip_mask is not 0, i.e., at least one bit is flipped. If no bit is flipped, use multi_qubit_Pauli_gate_Z_mask.
 * This function update the quantum state with Pauli operation. 
 * bit_flip_mask, phase_flip_mask, global_phase_90rot_count, and pivot_qubit_index must be computed before calling this function.
 * See get_masks_from_*_list for the above four arguemnts.
 */
//void multi_qubit_Pauli_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,UINT pivot_qubit_index, CPPCTYPE* state, ITYPE dim);
//void multi_qubit_Pauli_rotation_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, CPPCTYPE* state, ITYPE dim);
//void multi_qubit_Pauli_gate_Z_mask(ITYPE phase_flip_mask, CPPCTYPE* state, ITYPE dim);
//void multi_qubit_Pauli_rotation_gate_Z_mask(ITYPE phase_flip_mask, double angle, CPPCTYPE* state, ITYPE dim);

__device__ double atomicAdd_double_duplicate(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void penta_qubit_dense_matrix_gate_gpu(GTYPE *state_gpu, ITYPE dim){
	__shared__ GTYPE state_basis[1024];
    GTYPE tmp=make_cuDoubleComplex(0.0, 0.0);
	ITYPE loop_dim = dim >> 5;
	ITYPE basis = blockIdx.x * blockDim.x + threadIdx.x;
    
	int y;

	if (basis < loop_dim){
        for(y=0;y<5;++y) basis = insert_zero_to_basis_index_device(basis, sorted_insert_index_list_gpu[y] );
        for(y=0;y<5;++y) basis += (1ULL << target_index_list_gpu[y])*((threadIdx.y>>y)&1);
        
        state_basis[(threadIdx.x<<5)+threadIdx.y]=state_gpu[basis];
        __syncthreads();
        
        for(y=0;y<32;++y) tmp = cuCadd(tmp, cuCmul(matrix_const_gpu[(threadIdx.y<<5) + y], state_basis[(threadIdx.x<<5)+y] ));

        state_gpu[ basis ] = tmp;
	}
}

__host__ void penta_qubit_dense_matrix_gate_host(const unsigned int target_qubit_index[5], const CPPCTYPE matrix[1024], void* state, ITYPE dim, void* stream, UINT device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);

	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	checkCudaErrors(cudaMemcpyToSymbolAsync(matrix_const_gpu, matrix, sizeof(GTYPE)*1024, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);

	ITYPE loop_dim = dim >> 5;

    dim3 block;
    block.y = 32;
    block.x = loop_dim <= 32 ? loop_dim : 32;
	unsigned int grid = loop_dim / block.x;

	checkCudaErrors(cudaMemcpyToSymbolAsync(target_index_list_gpu, target_qubit_index, sizeof(UINT) * 5, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);

	unsigned int sort_list[5];
	memcpy(sort_list, target_qubit_index, sizeof(unsigned int) * 5);
	std::sort(sort_list, sort_list + 5);
	checkCudaErrors(cudaMemcpyToSymbolAsync(sorted_insert_index_list_gpu, sort_list, sizeof(UINT)*5, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
 
    penta_qubit_dense_matrix_gate_gpu<<< grid, block, 0, *cuda_stream>>>(state_gpu, dim);
    
	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__global__ void quad_qubit_dense_matrix_gate_shared_gpu(GTYPE *state_gpu, ITYPE dim){
	__shared__ GTYPE state_basis[1024];
    GTYPE tmp=make_cuDoubleComplex(0.0, 0.0);
	ITYPE loop_dim = dim >> 4;
	ITYPE basis = blockIdx.x * blockDim.x + threadIdx.x;
    
	int y;

	if (basis < loop_dim){
        for(y=0;y<4;++y) basis = insert_zero_to_basis_index_device(basis, sorted_insert_index_list_gpu[y] );
        for(y=0;y<4;++y) basis += (1ULL << sorted_insert_index_list_gpu[y])*((threadIdx.y>>y)&1);
        state_basis[(threadIdx.x<<4)+y]=state_gpu[basis];
        __syncthreads();
        
        for(y=0;y<16;++y) tmp = cuCadd(tmp, cuCmul(matrix_const_gpu[(threadIdx.y<<4) + y], state_basis[(threadIdx.x<<4)+threadIdx.y] ));

        state_gpu[ basis ] = tmp;
	}
}

__global__ void quad_qubit_dense_matrix_gate_gpu(unsigned int target0_qubit_index, unsigned int target1_qubit_index, unsigned int target2_qubit_index, unsigned int target3_qubit_index, 
												unsigned int sorted_index0, unsigned int sorted_index1, unsigned int sorted_index2, unsigned int sorted_index3, GTYPE *state_gpu, ITYPE dim){
	//ITYPE basis0;
	ITYPE basis[16];
	GTYPE d_buffer[16];
	ITYPE loop_dim = dim >> 4;
	ITYPE basis0 = blockIdx.x * blockDim.x + threadIdx.x;

	int x, y;

	if (basis0 < loop_dim){
        	//basis0 = j;
		// create base index
		basis0 = insert_zero_to_basis_index_device(basis0, sorted_index0 );
		basis0 = insert_zero_to_basis_index_device(basis0, sorted_index1 );
		basis0 = insert_zero_to_basis_index_device(basis0, sorted_index2 );
		basis0 = insert_zero_to_basis_index_device(basis0, sorted_index3);
		
		basis[0] =  basis0; // 0000
		basis[1] =  basis0 + (1ULL << target0_qubit_index); // 0001
		basis[2] = basis0 + (1ULL << target1_qubit_index); // 0010
		basis[3] = basis0 + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 0011
		basis[4] = basis0 + (1ULL << target2_qubit_index); // 0100
		basis[5] = basis0 + (1ULL << target2_qubit_index) + (1ULL << target0_qubit_index); // 0101
		basis[6] = basis0 + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index); // 0110
		basis[7] = basis0 + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 0111
		basis[8] = basis0 + (1ULL << target3_qubit_index); // 1000
		basis[9] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target0_qubit_index); // 1001
		basis[10] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target1_qubit_index); // 1010
		basis[11] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 1011
		basis[12] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target2_qubit_index); // 1100
		basis[13] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target2_qubit_index) + (1ULL << target0_qubit_index); // 1101
		basis[14] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index); // 1110
		basis[15] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 1111

		for(y = 0 ; y < 16 ; ++y ){
			d_buffer[y]=make_cuDoubleComplex(0.0,0.0);
			for(x = 0 ; x < 16 ; ++x){
				d_buffer[y] = cuCadd(d_buffer[y], 
						cuCmul(matrix_const_gpu[y*16 + x], state_gpu[ basis[x] ]));
			}
		}
        	for(y = 0 ; y < 16 ; ++y){
			state_gpu[basis[y]] = d_buffer[y];
		}
	}
}

__host__ void quad_qubit_dense_matrix_gate_host(const unsigned int target_qubit_index[4], const CPPCTYPE matrix[256], void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);
	
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	checkCudaErrors(cudaMemcpyToSymbolAsync(matrix_const_gpu, matrix, sizeof(GTYPE)*256, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
	ITYPE loop_dim = dim >> 4;

	unsigned int block = loop_dim <= 512 ? loop_dim : 512;
	unsigned int grid = loop_dim / block;
	unsigned int target0_qubit_index, target1_qubit_index, target2_qubit_index, target3_qubit_index;
	target0_qubit_index=target_qubit_index[0];
	target1_qubit_index=target_qubit_index[1];
	target2_qubit_index=target_qubit_index[2];
	target3_qubit_index=target_qubit_index[3];

	unsigned int sort_list[4];
	memcpy(sort_list, target_qubit_index, sizeof(unsigned int) * 4);
	std::sort(sort_list, sort_list + 4);

	quad_qubit_dense_matrix_gate_gpu << <grid, block, 0, *cuda_stream >> >(
		target0_qubit_index, target1_qubit_index, target2_qubit_index, target3_qubit_index, 
		sort_list[0], sort_list[1], sort_list[2], sort_list[3],
		state_gpu, dim
	);
	    
	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
    
    /*
    dim3 block;
    block.y = 16;
    block.x = loop_dim <= 64 ? loop_dim : 64;
	unsigned int grid = loop_dim / block.x;

	checkCudaErrors(cudaMemcpyToSymbol(sorted_insert_index_list_gpu, target_qubit_index, sizeof(UINT)*4), __FILE__, __LINE__);
    quad_qubit_dense_matrix_gate_shared_gpu << <grid, block >> >(state_gpu, dim);
	
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
    */
}

// target qubit 0 < target qubit 1 < target qubit 2
__global__ void triple_qubit_dense_matrix_gate_gpu(unsigned int target0_qubit_index, unsigned int target1_qubit_index, unsigned int target2_qubit_index, 
													unsigned int small, unsigned int mid, unsigned int large, GTYPE *state_gpu, ITYPE dim){
	ITYPE basis[8];
	GTYPE d_buffer[8];
	ITYPE loop_dim = dim >> 3;
	ITYPE basis0 = blockIdx.x * blockDim.x + threadIdx.x;

	int x, y;
	
	if (basis0 < loop_dim){
		// create base index
		basis0 = insert_zero_to_basis_index_device(basis0, small );
		basis0 = insert_zero_to_basis_index_device(basis0, mid );
		basis0 = insert_zero_to_basis_index_device(basis0, large );
		
		basis[0] =  basis0; // 000
		basis[1] =  basis0 + (1ULL << target0_qubit_index); // 001
		basis[2] = basis0 + (1ULL << target1_qubit_index); // 010
		basis[3] = basis0 + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 011
		basis[4] = basis0 + (1ULL << target2_qubit_index); // 100
		basis[5] = basis0 + (1ULL << target2_qubit_index) + (1ULL << target0_qubit_index); // 101
		basis[6] = basis0 + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index); // 110
		basis[7] = basis0 + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 111

		for(y = 0 ; y < 8 ; ++y ){
			d_buffer[y]=make_cuDoubleComplex(0.0,0.0);
			for(x = 0 ; x < 8 ; ++x){
				d_buffer[y] = cuCadd(d_buffer[y], 
						cuCmul(matrix_const_gpu[y*8 + x], state_gpu[ basis[x] ]));
			}
		}
        	for(y = 0 ; y < 8 ; ++y) state_gpu[basis[y]] = d_buffer[y];
	}
}

__host__ void triple_qubit_dense_matrix_gate_host(unsigned int target0_qubit_index, unsigned int target1_qubit_index, unsigned int target2_qubit_index, const CPPCTYPE matrix[64], void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);
	
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	
	checkCudaErrors(cudaMemcpyToSymbolAsync(matrix_const_gpu, matrix, sizeof(GTYPE)*64, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
	
    // (not using shared memory)
    ITYPE loop_dim = dim >> 3;
	unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;

	unsigned int small, mid, large, tmp;
	small = target0_qubit_index;
	mid = target1_qubit_index;
	large = target2_qubit_index;
	if (small > mid) { tmp = small; small = mid; mid = tmp; }
	if (mid > large) { tmp = large; large = mid; mid = tmp; }
	if (small > mid) { tmp = small; small = mid; mid = tmp; }

	triple_qubit_dense_matrix_gate_gpu << <grid, block, 0, *cuda_stream >> >(target0_qubit_index, target1_qubit_index, target2_qubit_index, small, mid, large, state_gpu, dim);
	
    checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

// target1 qubit index > target0 qubit index
__global__ void double_qubit_dense_matrix_gate_gpu(unsigned int target0_qubit_index, unsigned int target1_qubit_index,unsigned int small, unsigned int large, GTYPE *state_gpu, ITYPE dim){
	// unsigned int left, right;
	ITYPE head, body, tail, basis0;
	ITYPE basis[4];
	GTYPE d_buffer[4];
	ITYPE quad_dim = dim >> 2;
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;

	int x, y;
	/*
	if (target1_qubit_index > target2_qubit_index){
		left = target1_qubit_index;
		right = target2_qubit_index;
	}
	else {
		left = target2_qubit_index;
		right = target1_qubit_index;
	}
	*/
	// target1 qubit index > target2 qubit index
	
	if (j < quad_dim){
		head = j >> (large - 1);
		body = (j & ((1ULL << (large - 1)) - 1)) >> small; // (j % 2^(k-1)) >> i
		tail = j & ((1ULL << small) - 1); // j%(2^i)

		basis0 =  (head << (large + 1)) + (body << (small + 1)) + tail;
		basis[0] = basis0;
		basis[1] = basis0 + (1ULL << target0_qubit_index);
		basis[2] = basis0 + (1ULL << target1_qubit_index);
		basis[3] = basis0 + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index);

		for(y = 0 ; y < 4 ; ++y ){
			d_buffer[y]=make_cuDoubleComplex(0.0,0.0);
			for(x = 0 ; x < 4 ; ++x){
				d_buffer[y] = cuCadd(d_buffer[y], 
						cuCmul(matrix_const_gpu[y*4 + x], state_gpu[ basis[x] ]));
			}
		}
        for(y = 0 ; y < 4 ; ++y) state_gpu[basis[y]] = d_buffer[y];
	}
}

__host__ void double_qubit_dense_matrix_gate_host(unsigned int target0_qubit_index, unsigned int target1_qubit_index, const CPPCTYPE matrix[16], void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);
	
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;

	checkCudaErrors(cudaMemcpyToSymbolAsync(matrix_const_gpu, matrix, sizeof(GTYPE)*16, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
	ITYPE quad_dim = dim >> 2;
	unsigned int block = quad_dim <= 1024 ? quad_dim : 1024;
	unsigned int grid = quad_dim / block;

	unsigned int small;
	unsigned int large;
	small = (target0_qubit_index < target1_qubit_index) ? target0_qubit_index : target1_qubit_index;
	large = (target0_qubit_index < target1_qubit_index) ? target1_qubit_index : target0_qubit_index;

	double_qubit_dense_matrix_gate_gpu << <grid, block, 0, *cuda_stream >> >(target0_qubit_index, target1_qubit_index, small, large, state_gpu, dim);
	
    checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

// multi_qubit_PauliZ_gate
__device__ void multi_qubit_Pauli_gate_Z_mask_device(ITYPE phase_flip_mask, GTYPE* state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	// loop varaibles
	const ITYPE loop_dim = dim;
	if(state_index<loop_dim){
		// determine parity
		//UINT bit1_num = popcount64(state_index & phase_flip_mask);
		UINT bit1_num = __popcll(state_index & phase_flip_mask);
		// set values
		if(bit1_num&1) state_gpu[state_index] = make_cuDoubleComplex(-1*cuCreal(state_gpu[state_index]), -1*cuCimag(state_gpu[state_index]));
	}
}

__global__ void multi_qubit_Pauli_gate_Z_mask_gpu(ITYPE phase_flip_mask, GTYPE* state_gpu, ITYPE dim){
	multi_qubit_Pauli_gate_Z_mask_device(phase_flip_mask, state_gpu, dim);
}

__host__ void multi_qubit_Pauli_gate_Z_mask_host(ITYPE phase_flip_mask, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);
	
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	
    multi_qubit_Pauli_gate_Z_mask_gpu << <grid, block, 0, *cuda_stream >> >(phase_flip_mask, state_gpu, dim);		
	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__device__ void multi_qubit_Pauli_gate_XZ_mask_device(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, GTYPE* state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	// loop varaibles
	const ITYPE loop_dim = dim>>1;
	GTYPE PHASE_M90ROT[4] = { make_cuDoubleComplex(1.0,0.0), make_cuDoubleComplex(0.0,-1), make_cuDoubleComplex(-1,0.0), make_cuDoubleComplex(0.0,1)};

	if(state_index<loop_dim){
		// create base index
		ITYPE basis_0 = insert_zero_to_basis_index_device(state_index, pivot_qubit_index);

		// gather index
		ITYPE basis_1 = basis_0 ^ bit_flip_mask;

		// determine sign
		unsigned int sign_0 = __popcll(basis_0 & phase_flip_mask)&1;
		unsigned int sign_1 = __popcll(basis_1 & phase_flip_mask)&1;
		 
		// fetch values
		GTYPE cval_0 = state_gpu[basis_0];
		GTYPE cval_1 = state_gpu[basis_1];

		// set values
		state_gpu[basis_0] = cuCmul(cval_1, PHASE_M90ROT[(global_phase_90rot_count + sign_0*2)&3]); // a % 4 = a & (4-1)
		state_gpu[basis_1] = cuCmul(cval_0, PHASE_M90ROT[(global_phase_90rot_count + sign_1*2)&3]); // a % 4 = a & (4-1)
	}
}

__global__ void multi_qubit_Pauli_gate_XZ_mask_gpu(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, GTYPE* state_gpu, ITYPE dim){
	multi_qubit_Pauli_gate_XZ_mask_device(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_gpu, dim);
}

__host__ void multi_qubit_Pauli_gate_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	
    multi_qubit_Pauli_gate_XZ_mask_gpu<< <grid, block, 0, *cuda_stream >> >(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_gpu, dim);
	
	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__device__ void multi_qubit_Pauli_rotation_gate_XZ_mask_device(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	// loop varaibles
	ITYPE loop_dim = dim>>1;

	// coefs
	double cosval = cos(angle/2);
	double sinval = sin(angle/2);
	//GTYPE PHASE_90ROT[4] = {make_cuDoubleComplex(1.0,0.0), make_cuDoubleComplex(0.0,1.0), make_cuDoubleComplex(-1.0,0.0), make_cuDoubleComplex(0.0,-1.0)};
	GTYPE PHASE_M90ROT[4] = { make_cuDoubleComplex(1.0,0.0), make_cuDoubleComplex(0.0,-1), make_cuDoubleComplex(-1,0.0), make_cuDoubleComplex(0.0,1)};
	if(state_index<loop_dim){
		// create base index
		ITYPE basis_0 = insert_zero_to_basis_index_device(state_index, pivot_qubit_index);
		// gather index
		ITYPE basis_1 = basis_0 ^ bit_flip_mask;
		// determine parity
		unsigned int bit_parity_0 = __popcll(basis_0 & phase_flip_mask)&1;
		unsigned int bit_parity_1 = __popcll(basis_1 & phase_flip_mask)&1;
		
		// fetch values        
		GTYPE cval_0 = state_gpu[basis_0];
		GTYPE cval_1 = state_gpu[basis_1];
		
		// set values
		GTYPE tmp =  cuCmul(make_cuDoubleComplex(sinval*cuCreal(cval_1), sinval*cuCimag(cval_1)), PHASE_M90ROT[ (global_phase_90rot_count + bit_parity_0*2)&3 ]);
		//state[basis_0] = cuCmul(cosval, cval_0) + 1.i * sinval * cval_1 * PHASE_M90ROT[ (global_phase_90rot_count + bit_parity_0*2)&3 ]; // % 4
		state_gpu[basis_0] = cuCadd(make_cuDoubleComplex(cosval*cuCreal(cval_0), cosval*cuCimag(cval_0)), cuCmul(tmp, make_cuDoubleComplex(0.0,1.0)));
		
		//state[basis_1] = cosval * cval_1 + 1.i * sinval * cval_0 * PHASE_M90ROT[ (global_phase_90rot_count + bit_parity_1*2)&3 ]; // % 4
		tmp =  cuCmul(make_cuDoubleComplex(sinval*cuCreal(cval_0), sinval*cuCimag(cval_0)), PHASE_M90ROT[(global_phase_90rot_count + bit_parity_1*2)&3 ]);
		state_gpu[basis_1] = cuCadd(make_cuDoubleComplex(cosval*cuCreal(cval_1), cosval*cuCimag(cval_1)), cuCmul(tmp, make_cuDoubleComplex(0.0, 1.0)) ); // % 4
	}
}

__global__ void multi_qubit_Pauli_rotation_gate_XZ_mask_gpu(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim){
	multi_qubit_Pauli_rotation_gate_XZ_mask_device(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle, state_gpu, dim);
}

__host__ void multi_qubit_Pauli_rotation_gate_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	
	multi_qubit_Pauli_rotation_gate_XZ_mask_gpu<< <grid, block, 0, *cuda_stream >> >(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle, state_gpu, dim);
    
    checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__device__ void multi_qubit_Pauli_rotation_gate_Z_mask_device(ITYPE phase_flip_mask, double angle, GTYPE* state_gpu, ITYPE dim){
	// loop variables
	const ITYPE loop_dim = dim;
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	// coefs
	const double cosval = cos(angle/2);
	const double sinval = sin(angle/2);

	if(state_index<loop_dim){
		// determine sign
		UINT bit_parity = __popcll(state_index & phase_flip_mask)&1;
		int sign = 1 - 2*bit_parity;
		
		// set value
		state_gpu[state_index] = cuCmul(state_gpu[state_index], make_cuDoubleComplex(cosval, sign * sinval));
	}
}

__global__ void multi_qubit_Pauli_rotation_gate_Z_mask_gpu(ITYPE phase_flip_mask, double angle, GTYPE* state_gpu, ITYPE dim){
	multi_qubit_Pauli_rotation_gate_Z_mask_device(phase_flip_mask, angle, state_gpu, dim);
}

__host__ void multi_qubit_Pauli_rotation_gate_Z_mask_host(ITYPE phase_flip_mask, double angle, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	
	multi_qubit_Pauli_rotation_gate_Z_mask_gpu<< <grid, block, 0, *cuda_stream >> >(phase_flip_mask, angle, state_gpu, dim);
	
    checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__host__ void multi_qubit_Pauli_gate_partial_list_host(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	// create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_partial_list_gsim(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if (bit_flip_mask == 0) {
		multi_qubit_Pauli_gate_Z_mask_host(phase_flip_mask, state, dim, stream, device_number);
	}
	else {
		multi_qubit_Pauli_gate_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim, stream, device_number);
	}
}

__host__ void multi_qubit_Pauli_gate_whole_list_host(const UINT* Pauli_operator_type_list, UINT qubit_count, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);
	// create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_whole_list_gsim(Pauli_operator_type_list, qubit_count, &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if (bit_flip_mask == 0) {
		multi_qubit_Pauli_gate_Z_mask_host(phase_flip_mask, state, dim, stream, device_number);
	}
	else {
		multi_qubit_Pauli_gate_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim, stream, device_number);
	}
}

__host__ void multi_qubit_Pauli_rotation_gate_partial_list_host(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, double angle, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	// create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_partial_list_gsim(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if(bit_flip_mask == 0){
		multi_qubit_Pauli_rotation_gate_Z_mask_host(phase_flip_mask, angle, state, dim, stream, device_number);
	}else{
		multi_qubit_Pauli_rotation_gate_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index,angle, state, dim, stream, device_number);
	}
}

__host__ void multi_qubit_Pauli_rotation_gate_whole_list_host(const UINT* Pauli_operator_type_list, UINT qubit_count, double angle, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	// create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_whole_list_gsim(Pauli_operator_type_list, qubit_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if (bit_flip_mask == 0) {
		multi_qubit_Pauli_rotation_gate_Z_mask_host(phase_flip_mask, angle, state, dim, stream, device_number);
	}
	else {
		multi_qubit_Pauli_rotation_gate_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle, state, dim, stream, device_number);
	}
}

// target_qubit_count <= 5
__global__ void multi_qubit_dense_matrix_gate_shared_gpu(UINT target_qubit_index_count, GTYPE *state_gpu, ITYPE dim){
	__shared__ GTYPE state_basis[1024];
    GTYPE tmp=make_cuDoubleComplex(0.0, 0.0);
	ITYPE loop_dim = dim >> target_qubit_index_count;
	ITYPE basis = blockIdx.x * blockDim.x + threadIdx.x;
    
	int j;
    
    ITYPE mat_len = 1ULL << target_qubit_index_count;
	if (basis < loop_dim){
        for(j=0;j<target_qubit_index_count;++j) basis = insert_zero_to_basis_index_device(basis, sorted_insert_index_list_gpu[j] );
        for(j=0;j<target_qubit_index_count;++j) basis += (1ULL << target_index_list_gpu[j])*((threadIdx.y>>j)&1);
        state_basis[(threadIdx.x<<target_qubit_index_count)+threadIdx.y]=state_gpu[basis];
        __syncthreads();
        
        for(j=0;j<mat_len;++j) tmp = cuCadd(tmp, cuCmul(matrix_const_gpu[(threadIdx.y<<target_qubit_index_count) + j], state_basis[(threadIdx.x<<target_qubit_index_count)+j] ));

        state_gpu[ basis ] = tmp;
	}
}

// target_qubit_count <= 10
__global__ void multi_qubit_dense_matrix_gate_shared_gpu(UINT target_qubit_index_count, GTYPE* matrix_gpu, GTYPE *state_gpu, ITYPE dim){
	__shared__ GTYPE state_basis[1024];
    GTYPE tmp=make_cuDoubleComplex(0.0, 0.0);
	ITYPE loop_dim = dim >> target_qubit_index_count;
	ITYPE basis = blockIdx.x * blockDim.x + threadIdx.x;
    
	int j;
    
    ITYPE mat_len = 1ULL << target_qubit_index_count;
	if (basis < loop_dim){
        for(j=0;j<target_qubit_index_count;++j) basis = insert_zero_to_basis_index_device(basis, sorted_insert_index_list_gpu[j] );
        for(j=0;j<target_qubit_index_count;++j) basis += (1ULL << target_index_list_gpu[j])*((threadIdx.y>>j)&1);
        state_basis[(threadIdx.x<<target_qubit_index_count)+threadIdx.y]=state_gpu[basis];
        __syncthreads();
        
        for(j=0;j<mat_len;++j) tmp = cuCadd(tmp, cuCmul(matrix_gpu[(threadIdx.y<<target_qubit_index_count) + j], state_basis[(threadIdx.x<<target_qubit_index_count)+j] ));

        state_gpu[ basis ] = tmp;
	}
}

// there is no atomicAdd
// target_qubit_index_count<=11
__global__ void multi_qubit_dense_matrix_gate_half_shared_gpu(UINT target_qubit_index_count, GTYPE* matrix_gpu, GTYPE *state_gpu, ITYPE dim){
	__shared__ GTYPE state_basis[2048];
	ITYPE loop_dim = dim >> target_qubit_index_count;
	ITYPE basis = blockIdx.x * blockDim.x + threadIdx.x;
    ITYPE basis0, basis1;
    
    ITYPE matrix_len = 1ULL << target_qubit_index_count;
    //ITYPE half_matrix_len = 1ULL << (target_qubit_index_count-1);
	if (basis < loop_dim){
        for(int j=0;j<target_qubit_index_count;++j) basis = insert_zero_to_basis_index_device(basis, sorted_insert_index_list_gpu[j] );
        for(int j=0;j<target_qubit_index_count-1;++j) basis += (1ULL << target_index_list_gpu[j+1])*((threadIdx.y>>j)&1);
        basis0=basis;
        basis1=basis0^(1ULL<<sorted_insert_index_list_gpu[0]);
        state_basis[(threadIdx.x<<target_qubit_index_count)+(threadIdx.y<<1)]=state_gpu[basis0];
        state_basis[(threadIdx.x<<target_qubit_index_count)+(threadIdx.y<<1)+1]=state_gpu[basis1];
        __syncthreads();
        
        GTYPE d_buff = make_cuDoubleComplex(0.0, 0.0);
        for(int j=0;j<matrix_len;++j) d_buff = cuCadd(d_buff, cuCmul(matrix_gpu[((threadIdx.y<<1)<<target_qubit_index_count) + j], state_basis[(threadIdx.x<<target_qubit_index_count)+j] ));
        state_gpu[ basis0 ] = d_buff;
        
        d_buff = make_cuDoubleComplex(0.0, 0.0);
        for(int j=0;j<matrix_len;++j) d_buff = cuCadd(d_buff, cuCmul(matrix_gpu[(((threadIdx.y<<1)+1)<<target_qubit_index_count) + j], state_basis[(threadIdx.x<<target_qubit_index_count)+j] ));
        state_gpu[ basis1 ] = d_buff;
        // printf("basis0: %d, basis1: %d\n", (int)basis0, (int)basis1);
	}
}

__global__ void multi_qubit_dense_matrix_gate_gpu(UINT target_qubit_index_count, GTYPE* matrix_gpu, GTYPE* state_gpu, GTYPE* state_gpu_copy, ITYPE dim) {
    __shared__ GTYPE state_basis[1024];
	ITYPE loop_dim = dim >> target_qubit_index_count;
    
    ITYPE large_block_index = 0;
    ITYPE large_block_residual = 0;
    ITYPE block_loop_dim = 1; //target_qubit_index_count-3;
    ITYPE block_index = 0;
    ITYPE block_residual = 0; //block_loop_dim<=1 ? 0 : blockIdx.x % (1ULL<<block_loop_dim);
	ITYPE basis = blockIdx.x * blockDim.x + threadIdx.x;
    ITYPE assign_basis;
    ITYPE basis0;

    if(target_qubit_index_count>=10+1){
        block_loop_dim = 1ULL << (target_qubit_index_count-10);
        large_block_index = blockIdx.x / (block_loop_dim*block_loop_dim);
        large_block_residual = blockIdx.x % (block_loop_dim*block_loop_dim);
        block_index= large_block_residual / block_loop_dim;
        block_residual = blockIdx.x % block_loop_dim;
        basis = large_block_index * blockDim.x + threadIdx.x;
    }
    
    ITYPE matrix_len = 1ULL << target_qubit_index_count;
    if(basis < loop_dim){
        ITYPE tmp = (block_residual<<10) + threadIdx.y;
        for(int j=0;j<target_qubit_index_count;++j) basis = insert_zero_to_basis_index_device(basis, sorted_insert_index_list_gpu[j] );
        basis0=basis;
        for(int j=0;j<target_qubit_index_count;++j) basis += (1ULL << target_index_list_gpu[j])*( (tmp>>j) & 1);
        state_basis[(threadIdx.x<<target_qubit_index_count)+threadIdx.y]=state_gpu_copy[basis];
        if(target_qubit_index_count>=10+1){
            tmp = (block_index << 10) + threadIdx.y;
            assign_basis = basis0;
            for(int j=0;j<target_qubit_index_count;++j) assign_basis += (1ULL << target_index_list_gpu[j])*( (tmp>>j) & 1);
        }else{
            assign_basis = basis;
        }
        __syncthreads();

        GTYPE d_buff = make_cuDoubleComplex(0.0, 0.0);
        ITYPE tmp_len = block_residual << 10;
        if(matrix_len>1024) matrix_len=1024;
        ITYPE row_index = ( block_index << 10 ) + threadIdx.y;
        for(ITYPE j=0;j<matrix_len;++j) d_buff = cuCadd(d_buff, cuCmul(matrix_gpu[(row_index<<target_qubit_index_count) + j + tmp_len], state_basis[(threadIdx.x<<target_qubit_index_count)+j] ));
		atomicAdd_double_duplicate(&(state_gpu[assign_basis].x), d_buff.x);
		atomicAdd_double_duplicate(&(state_gpu[assign_basis].y), d_buff.y);
    }
}

__host__ void multi_qubit_dense_matrix_gate_small_qubit_host(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CPPCTYPE* matrix, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);

	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;

	// matrix dim, mask, buffer
	ITYPE matrix_dim = 1ULL << target_qubit_index_count;

	// insert index
	UINT* h_sorted_insert_index_list = create_sorted_ui_list_gsim(target_qubit_index_list, target_qubit_index_count);

	// loop variables
	ITYPE loop_dim = dim >> target_qubit_index_count;

	GTYPE* matrix_gpu;

	unsigned int max_block_size = 1024 / matrix_dim;
	dim3 block;
	block.y = matrix_dim;
	block.x = loop_dim <= max_block_size ? loop_dim : max_block_size;
	unsigned int grid = loop_dim / block.x;

	if (target_qubit_index_count <= 5) {
		checkCudaErrors(cudaMemcpyToSymbolAsync(matrix_const_gpu, matrix, sizeof(GTYPE)*matrix_dim*matrix_dim, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		checkCudaErrors(cudaMemcpyToSymbolAsync(target_index_list_gpu, target_qubit_index_list, sizeof(UINT)*target_qubit_index_count, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		checkCudaErrors(cudaMemcpyToSymbolAsync(sorted_insert_index_list_gpu, h_sorted_insert_index_list, sizeof(UINT)*target_qubit_index_count, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);

		multi_qubit_dense_matrix_gate_shared_gpu << <grid, block, 0, *cuda_stream >> > (target_qubit_index_count, state_gpu, dim);
	}
	else if (target_qubit_index_count <= 10) {
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&matrix_gpu), matrix_dim *matrix_dim * sizeof(GTYPE)), __FILE__, __LINE__);
		checkCudaErrors(cudaMemcpyAsync(matrix_gpu, matrix, matrix_dim *matrix_dim * sizeof(GTYPE), cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		checkCudaErrors(cudaMemcpyToSymbolAsync(target_index_list_gpu, target_qubit_index_list, sizeof(UINT)*target_qubit_index_count, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		checkCudaErrors(cudaMemcpyToSymbolAsync(sorted_insert_index_list_gpu, h_sorted_insert_index_list, sizeof(UINT)*target_qubit_index_count, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);

		multi_qubit_dense_matrix_gate_shared_gpu << <grid, block, 0, *cuda_stream >> > (target_qubit_index_count, matrix_gpu, state_gpu, dim);
	}
	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);

	if (target_qubit_index_count > 5) cudaFree(matrix_gpu);
	free((UINT*)h_sorted_insert_index_list);

	state = reinterpret_cast<void*>(state_gpu);
}

__host__ void multi_qubit_dense_matrix_gate_11qubit_host(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CPPCTYPE* matrix, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);
	
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	//cudaError cudaStatus;

	// matrix dim, mask, buffer
    ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    UINT* h_sorted_insert_index_list = create_sorted_ui_list_gsim(target_qubit_index_list, target_qubit_index_count);
    
    // loop variables
	//ITYPE loop_dim = dim >> target_qubit_index_count;
	
    GTYPE *matrix_gpu;
    
    dim3 block;
    block.y = (matrix_dim>>1) <= 1024 ? (matrix_dim>>1) : 1024;
    unsigned int max_block_size = 1024 / block.y;
    block.x = dim/block.y <= max_block_size ? dim/block.y : max_block_size;
    unsigned int grid = dim / block.x / block.y;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&matrix_gpu), matrix_dim *matrix_dim * sizeof(GTYPE) ), __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpyAsync(matrix_gpu, matrix, matrix_dim *matrix_dim * sizeof(GTYPE), cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpyToSymbolAsync(target_index_list_gpu, target_qubit_index_list, sizeof(UINT)*target_qubit_index_count, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpyToSymbolAsync(sorted_insert_index_list_gpu, h_sorted_insert_index_list, sizeof(UINT)*target_qubit_index_count, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
            
    multi_qubit_dense_matrix_gate_half_shared_gpu << <grid, block, 0, *cuda_stream >> >(target_qubit_index_count, matrix_gpu, state_gpu, dim);
    
    checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
    checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
    checkCudaErrors(cudaFree(matrix_gpu), __FILE__, __LINE__);
    
    free((UINT*)h_sorted_insert_index_list);
    state = reinterpret_cast<void*>(state_gpu);
}

__host__ void multi_qubit_dense_matrix_gate_more_than_11qubit_host(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CPPCTYPE* matrix, void* state, ITYPE dim, void* stream, UINT device_number){
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);

	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	//cudaError cudaStatus;

	// matrix dim, mask, buffer
    ITYPE matrix_dim = 1ULL << target_qubit_index_count;

    UINT* h_sorted_insert_index_list = create_sorted_ui_list_gsim(target_qubit_index_list, target_qubit_index_count);
    
    // loop variables
	ITYPE loop_dim = dim >> target_qubit_index_count;
	
    GTYPE *matrix_gpu;
    dim3 grid, block;
    block.y = matrix_dim <= (1ULL<<10) ? matrix_dim : (1ULL<<10);
    unsigned int max_block_size = (1ULL<<10) / block.y;
    block.x = dim/block.y <= max_block_size ? dim/block.y : max_block_size;
    grid.x = dim / block.x / block.y;
    if(target_qubit_index_count>=10+1) grid.x = (1ULL<<((target_qubit_index_count-10)<<1)) * loop_dim;
    
    GTYPE* state_gpu_copy;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&matrix_gpu), matrix_dim *matrix_dim * sizeof(GTYPE) ), __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpyAsync(matrix_gpu, matrix, matrix_dim *matrix_dim * sizeof(GTYPE), cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpyToSymbolAsync(target_index_list_gpu, target_qubit_index_list, sizeof(UINT)*target_qubit_index_count, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpyToSymbolAsync(sorted_insert_index_list_gpu, h_sorted_insert_index_list, sizeof(UINT)*target_qubit_index_count, 0,cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&state_gpu_copy), dim * sizeof(GTYPE) ), __FILE__, __LINE__);
    checkCudaErrors(cudaMemcpyAsync(state_gpu_copy, state_gpu, dim * sizeof(GTYPE), cudaMemcpyDeviceToDevice, *cuda_stream), __FILE__, __LINE__);
    checkCudaErrors(cudaMemsetAsync(state_gpu, 0, dim * sizeof(GTYPE), *cuda_stream), __FILE__, __LINE__);
    
    multi_qubit_dense_matrix_gate_gpu<<< grid, block, 0, *cuda_stream >>>(target_qubit_index_count, matrix_gpu, state_gpu, state_gpu_copy, dim);
    
    checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
    checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
    
    cudaFree(state_gpu_copy);
    cudaFree(matrix_gpu);
    free((UINT*)h_sorted_insert_index_list);
    state = reinterpret_cast<void*>(state_gpu);
}

__host__ void multi_qubit_dense_matrix_gate_host(const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CPPCTYPE* matrix, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	if (target_qubit_index_count == 1) {
		single_qubit_dense_matrix_gate_host(target_qubit_index_list[0], matrix, state, dim, stream, device_number);
	}
	else if (target_qubit_index_count == 2) {
		double_qubit_dense_matrix_gate_host(target_qubit_index_list[0], target_qubit_index_list[1], matrix, state, dim, stream, device_number);
	}
	else if (target_qubit_index_count == 3) {
		triple_qubit_dense_matrix_gate_host(target_qubit_index_list[0], target_qubit_index_list[1], target_qubit_index_list[2], matrix, state, dim, stream, device_number);
	}
	else if (target_qubit_index_count == 4) {
		UINT target_qubit_index_list_copy[4];
		for (int i = 0; i < 4; ++i) target_qubit_index_list_copy[i] = target_qubit_index_list[i];
		quad_qubit_dense_matrix_gate_host(target_qubit_index_list_copy, matrix, state, dim, stream, device_number);
	}
	else if (target_qubit_index_count == 11) {
		multi_qubit_dense_matrix_gate_11qubit_host(target_qubit_index_list, target_qubit_index_count, matrix, state, dim, stream, device_number);
	}
	else if (target_qubit_index_count >= 12) {
		multi_qubit_dense_matrix_gate_more_than_11qubit_host(target_qubit_index_list, target_qubit_index_count, matrix, state, dim, stream, device_number);
	}
	else {
		multi_qubit_dense_matrix_gate_small_qubit_host(target_qubit_index_list, target_qubit_index_count, matrix, state, dim, stream, device_number);
	}
}

// target_qubit_index_count <= 5
__global__ void single_qubit_control_multi_qubit_dense_matrix_gate_const_gpu(UINT control_qubit_index, UINT control_value, UINT target_qubit_index_count, GTYPE* state, ITYPE dim) {

    // control mask
    const ITYPE control_mask = (1ULL << control_qubit_index) * control_value;
    const UINT insert_index_count = target_qubit_index_count + 1;
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;

    // loop varaibles
    const ITYPE loop_dim = dim >> insert_index_count;
    GTYPE d_buffer[1024];
    ITYPE state_index  = blockIdx.x * blockDim.x + threadIdx.x;

    if(state_index < loop_dim){
        
        // create base index
        ITYPE basis_0 = state_index;
        for(UINT cursor=0; cursor < insert_index_count ; cursor++){
            UINT insert_index = sorted_insert_index_list_gpu[cursor];
            basis_0 = insert_zero_to_basis_index_device(basis_0, insert_index );
        }

        // flip control
        basis_0 ^= control_mask;

       // compute matrix mul
       for(ITYPE y = 0 ; y < matrix_dim ; ++y ){
            d_buffer[y]=make_cuDoubleComplex(0.0, 0.0);
            for(ITYPE x = 0 ; x < matrix_dim ; ++x){
                d_buffer[y] = cuCadd( d_buffer[y], cuCmul( matrix_const_gpu[y*matrix_dim + x], state[ basis_0 ^ matrix_mask_list_gpu[x] ]));
			}
        }

        // set result
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            state[basis_0 ^ matrix_mask_list_gpu[y]] = d_buffer[y];
        }
    }
}

// target_qubit_index_count <= 10
__global__ void single_qubit_control_multi_qubit_dense_matrix_gate_const_gpu(UINT control_qubit_index, UINT control_value, UINT target_qubit_index_count, const GTYPE* matrix, GTYPE* state, ITYPE dim) {

    // control mask
    const ITYPE control_mask = (1ULL << control_qubit_index) * control_value;
    const UINT insert_index_count = target_qubit_index_count + 1;
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;

    // loop varaibles
    const ITYPE loop_dim = dim >> insert_index_count;
    GTYPE d_buffer[1024];
    ITYPE state_index  = blockIdx.x * blockDim.x + threadIdx.x;

    if(state_index < loop_dim){
        
        // create base index
        ITYPE basis_0 = state_index;
        for(UINT cursor=0; cursor < insert_index_count ; cursor++){
            UINT insert_index = sorted_insert_index_list_gpu[cursor];
            basis_0 = insert_zero_to_basis_index_device(basis_0, insert_index );
        }

        // flip control
        basis_0 ^= control_mask;

       // compute matrix mul
       for(ITYPE y = 0 ; y < matrix_dim ; ++y ){
            d_buffer[y]=make_cuDoubleComplex(0.0, 0.0);
            for(ITYPE x = 0 ; x < matrix_dim ; ++x){
                d_buffer[y] = cuCadd(d_buffer[y] , cuCmul( matrix[y*matrix_dim + x], state[ basis_0 ^ matrix_mask_list_gpu[x] ]));
			}
        }

        // set result
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            state[basis_0 ^ matrix_mask_list_gpu[y]] = d_buffer[y];
        }
    }
}

__host__ void single_qubit_control_multi_qubit_dense_matrix_gate_host(UINT control_qubit_index, UINT control_value, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CPPCTYPE* matrix, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);
	
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
    // matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    ITYPE* matrix_mask_list = create_matrix_mask_list_gsim(target_qubit_index_list, target_qubit_index_count);

    // insert list
    const UINT insert_index_count = target_qubit_index_count + 1;
    UINT* sorted_insert_index_list = create_sorted_ui_list_value_gsim(target_qubit_index_list, target_qubit_index_count ,control_qubit_index);

    GTYPE *d_matrix, *d_matrix_mask_list;

    // loop varaibles
    const ITYPE loop_dim = dim >> insert_index_count;

	unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;
    
    if(target_qubit_index_count<=10){
        if(target_qubit_index_count>=3){
            unsigned int tmp_block = 1ULL << (13-target_qubit_index_count);
            block = loop_dim <= tmp_block ? loop_dim : tmp_block;
        }else{
            block = loop_dim <= 1024 ? loop_dim : 1024;
	    }
        grid = loop_dim / block;
        
        if(target_qubit_index_count<=5){
		    checkCudaErrors(cudaMemcpyToSymbolAsync(matrix_const_gpu, matrix, sizeof(GTYPE)*matrix_dim*matrix_dim, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbolAsync(matrix_mask_list_gpu, matrix_mask_list, sizeof(ITYPE)*matrix_dim, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbolAsync(sorted_insert_index_list_gpu, sorted_insert_index_list, sizeof(UINT)*(target_qubit_index_count + 1), 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);

            single_qubit_control_multi_qubit_dense_matrix_gate_const_gpu<<< grid, block, 0, *cuda_stream >>> (control_qubit_index, control_value, target_qubit_index_count, state_gpu, dim);
        }else{
		    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_matrix), matrix_dim *matrix_dim * sizeof(GTYPE) ), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyAsync(d_matrix, matrix, matrix_dim *matrix_dim * sizeof(GTYPE), cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_matrix_mask_list), matrix_dim *matrix_dim * sizeof(GTYPE) ), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyAsync(d_matrix_mask_list, matrix_mask_list, sizeof(ITYPE)*matrix_dim, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbolAsync(sorted_insert_index_list_gpu, sorted_insert_index_list, sizeof(UINT)*(target_qubit_index_count + 1), 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		    
            single_qubit_control_multi_qubit_dense_matrix_gate_const_gpu<<< grid, block, 0, *cuda_stream >>> (control_qubit_index, control_value, target_qubit_index_count, d_matrix, state_gpu, dim);
        }
	}else{
        printf("The max number of targets is limited to 10.");
        assert(0);
	}

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
    
    // Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus, __FILE__, __LINE__);
    
    if(target_qubit_index_count>5){
        cudaFree(d_matrix);
        cudaFree(d_matrix_mask_list);
    }
    free(sorted_insert_index_list);
    free(matrix_mask_list);
 
    state = reinterpret_cast<void*>(state_gpu);
}

// target_qubit_index_count <= 5
__global__ void multi_qubit_control_multi_qubit_dense_matrix_gate_const_gpu(ITYPE control_mask, UINT target_qubit_index_count, ITYPE control_qubit_index_count, GTYPE* state, ITYPE dim) {

    // control mask
    const UINT insert_index_count = target_qubit_index_count + control_qubit_index_count;
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;

    // loop varaibles
    const ITYPE loop_dim = dim >> insert_index_count;
    GTYPE d_buffer[1024];
    ITYPE state_index  = blockIdx.x * blockDim.x + threadIdx.x;

    if(state_index < loop_dim){

        // create base index
        ITYPE basis_0 = state_index;
        for(UINT cursor=0; cursor < insert_index_count ; cursor++){
            UINT insert_index = sorted_insert_index_list_gpu[cursor];
            basis_0 = insert_zero_to_basis_index_device(basis_0, insert_index );
        }

        // flip control
        basis_0 ^= control_mask;

       // compute matrix mul
       for(ITYPE y = 0 ; y < matrix_dim ; ++y ){
            d_buffer[y]=make_cuDoubleComplex(0.0, 0.0);
            for(ITYPE x = 0 ; x < matrix_dim ; ++x){
                d_buffer[y] = cuCadd(d_buffer[y] , cuCmul( matrix_const_gpu[y*matrix_dim + x], state[ basis_0 ^ matrix_mask_list_gpu[x] ]));
			}
        }

        // set result
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            state[basis_0 ^ matrix_mask_list_gpu[y]] = d_buffer[y];
        }
    }
}

// target_qubit_index_count <= 10
__global__ void multi_qubit_control_multi_qubit_dense_matrix_gate_const_gpu(ITYPE control_mask, UINT target_qubit_index_count, ITYPE control_qubit_index_count, const GTYPE* matrix, GTYPE* state, ITYPE dim) {

    // control mask
    const UINT insert_index_count = target_qubit_index_count + control_qubit_index_count;
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;

    // loop varaibles
    const ITYPE loop_dim = dim >> insert_index_count;
    GTYPE d_buffer[1024];
    ITYPE state_index  = blockIdx.x * blockDim.x + threadIdx.x;

    if(state_index < loop_dim){

        // create base index
        ITYPE basis_0 = state_index;
        for(UINT cursor=0; cursor < insert_index_count ; cursor++){
            UINT insert_index = sorted_insert_index_list_gpu[cursor];
            basis_0 = insert_zero_to_basis_index_device(basis_0, insert_index );
        }

        // flip control
        basis_0 ^= control_mask;

       // compute matrix mul
       for(ITYPE y = 0 ; y < matrix_dim ; ++y ){
            d_buffer[y]=make_cuDoubleComplex(0.0, 0.0);
            for(ITYPE x = 0 ; x < matrix_dim ; ++x){
                d_buffer[y] = cuCadd(d_buffer[y] , cuCmul( matrix[y*matrix_dim + x], state[ basis_0 ^ matrix_mask_list_gpu[x] ]));
			}
        }

        // set result
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            state[basis_0 ^ matrix_mask_list_gpu[y]] = d_buffer[y];
        }
    }
}

__host__ void multi_qubit_control_multi_qubit_dense_matrix_gate_host(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CPPCTYPE* matrix, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);

	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*> (stream);
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;

    // matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    ITYPE* matrix_mask_list = create_matrix_mask_list_gsim(target_qubit_index_list, target_qubit_index_count);

    // insert index
    UINT* sorted_insert_index_list = create_sorted_ui_list_list_gsim(target_qubit_index_list, target_qubit_index_count, control_qubit_index_list, control_qubit_index_count);
    
    // control mask
	ITYPE control_mask = create_control_mask_gsim(control_qubit_index_list, control_value_list, control_qubit_index_count);
    
    // loop varaibles
    const ITYPE loop_dim = dim >> (target_qubit_index_count+control_qubit_index_count);

    GTYPE *d_matrix, *d_matrix_mask_list;

	unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;
    
    if(target_qubit_index_count<=10){
        if(target_qubit_index_count>=3){
            unsigned int tmp_block = 1ULL << (13-target_qubit_index_count);
            block = loop_dim <= tmp_block ? loop_dim : tmp_block;
        }else{
            block = loop_dim <= 1024 ? loop_dim : 1024;
	    }
        grid = loop_dim / block;
        
        if(target_qubit_index_count<=5){
		    checkCudaErrors(cudaMemcpyToSymbolAsync(matrix_const_gpu, matrix, sizeof(GTYPE)*matrix_dim*matrix_dim, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbolAsync(matrix_mask_list_gpu, matrix_mask_list, sizeof(ITYPE)*matrix_dim, 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbolAsync(sorted_insert_index_list_gpu, sorted_insert_index_list, sizeof(UINT)*(target_qubit_index_count+control_qubit_index_count), 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);

            multi_qubit_control_multi_qubit_dense_matrix_gate_const_gpu<<< grid, block, 0, *cuda_stream >>> (control_mask, target_qubit_index_count, control_qubit_index_count, state_gpu, dim);
        }else{
		    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_matrix), matrix_dim *matrix_dim * sizeof(GTYPE) ), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyAsync(d_matrix, matrix, matrix_dim *matrix_dim * sizeof(GTYPE), cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_matrix_mask_list), matrix_dim *matrix_dim * sizeof(GTYPE) ), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyAsync(d_matrix_mask_list, matrix_mask_list, sizeof(ITYPE)*matrix_dim, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbolAsync(sorted_insert_index_list_gpu, sorted_insert_index_list, sizeof(UINT)*(target_qubit_index_count + control_qubit_index_count), 0, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
		    
            multi_qubit_control_multi_qubit_dense_matrix_gate_const_gpu<<< grid, block, 0, *cuda_stream >>> (control_mask, target_qubit_index_count, control_qubit_index_count, d_matrix, state_gpu, dim);
        }
	}else{
        printf("The max number of targets is limited to 10.");
        assert(0);
	}

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
    
    // Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus, __FILE__, __LINE__);
    
    if(target_qubit_index_count>5){
        cudaFree(d_matrix);
        cudaFree(d_matrix_mask_list);
    }
    free(sorted_insert_index_list);
    free(matrix_mask_list);
 
    state = reinterpret_cast<void*>(state_gpu);
}

// n_qubit <= 10
__global__ void multi_qubit_diagonal_matrix_gate_gpu(GTYPE* state_gpu, ITYPE dim) {
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (state_index < dim) {
		state_gpu[state_index] = cuCmul(matrix_const_gpu[state_index], state_gpu[state_index]);
	}
}

// n_qubit > 10
__global__ void multi_qubit_diagonal_matrix_gate_gpu(GTYPE* matrix_gpu, GTYPE* state_gpu, ITYPE dim) {
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (state_index < dim) {
		state_gpu[state_index] = cuCmul(matrix_gpu[state_index], state_gpu[state_index]);
	}
}

__host__ void multi_qubit_diagonal_matrix_gate_with_constant_memory_host(const CPPCTYPE* diagonal_matrix, void* state, ITYPE dim, void* stream, UINT device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	checkCudaErrors(cudaMemcpyToSymbol(matrix_const_gpu, diagonal_matrix, sizeof(GTYPE) * dim), __FILE__, __LINE__);

	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;

	multi_qubit_diagonal_matrix_gate_gpu << <grid, block, 0, * cuda_stream >> > (state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__host__ void multi_qubit_diagonal_matrix_gate_with_global_memory_host(const CPPCTYPE* diagonal_matrix, void* state, ITYPE dim, void* stream, UINT device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	GTYPE* d_matrix;
	checkCudaErrors(cudaMalloc((void**)& d_matrix, sizeof(GTYPE) * dim), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpyAsync(d_matrix, diagonal_matrix, sizeof(GTYPE) * dim, cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);

	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;

	multi_qubit_diagonal_matrix_gate_gpu << <grid, block, 0, * cuda_stream >> > (d_matrix, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
	cudaFree(d_matrix);
	state = reinterpret_cast<void*>(state_gpu);
}

__host__ void multi_qubit_diagonal_matrix_gate_host(const CPPCTYPE* diagonal_matrix, void* state, ITYPE dim, void* stream, UINT device_number) {
	if (dim <= 1024) {
		multi_qubit_diagonal_matrix_gate_with_constant_memory_host(diagonal_matrix, state, dim, stream, device_number);
	}
	else {
		multi_qubit_diagonal_matrix_gate_with_global_memory_host(diagonal_matrix, state, dim, stream, device_number);
	}
}

