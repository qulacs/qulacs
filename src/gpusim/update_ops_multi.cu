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

// maximum # of GTYPE elements allocating on constant memory: 4096 
__constant__ GTYPE matrix_const_gpu[1024];
__constant__ ITYPE matrix_mask_list_gpu[1024];
__constant__ UINT sorted_insert_index_list_gpu[10];

/**  vqcsim からの移植
 * perform multi_qubit_Pauli_gate with XZ mask.
 * 
 * This function assumes bit_flip_mask is not 0, i.e., at least one bit is flipped. If no bit is flipped, use multi_qubit_Pauli_gate_Z_mask.
 * This function update the quantum state with Pauli operation. 
 * bit_flip_mask, phase_flip_mask, global_phase_90rot_count, and pivot_qubit_index must be computed before calling this function.
 * See get_masks_from_*_list for the above four arguemnts.
 */
void multi_qubit_Pauli_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,UINT pivot_qubit_index, CPPCTYPE* state, ITYPE dim);
void multi_qubit_Pauli_rotation_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, CPPCTYPE* state, ITYPE dim);
void multi_qubit_Pauli_gate_Z_mask(ITYPE phase_flip_mask, CPPCTYPE* state, ITYPE dim);
void multi_qubit_Pauli_rotation_gate_Z_mask(ITYPE phase_flip_mask, double angle, CPPCTYPE* state, ITYPE dim);

//penta
__global__ void penta_qubit_dense_matrix_gate_gpu(unsigned int target4_qubit_index, unsigned int target3_qubit_index, unsigned int target2_qubit_index, unsigned int target1_qubit_index, unsigned int target0_qubit_index, GTYPE *state_gpu, ITYPE dim){
	//ITYPE basis0;
	ITYPE basis[32];
	GTYPE d_buffer[32];
	ITYPE loop_dim = dim >> 5;
	ITYPE basis0 = blockIdx.x * blockDim.x + threadIdx.x;

	int x, y;

	if (basis0 < loop_dim){
        	//basis0 = j;
		// create base index
		basis0 = insert_zero_to_basis_index_device(basis0, target0_qubit_index );
		basis0 = insert_zero_to_basis_index_device(basis0, target1_qubit_index );
		basis0 = insert_zero_to_basis_index_device(basis0, target2_qubit_index );
		basis0 = insert_zero_to_basis_index_device(basis0, target3_qubit_index );
		basis0 = insert_zero_to_basis_index_device(basis0, target4_qubit_index );
		
		basis[0] =  basis0; // 00000
		basis[1] =  basis0 + (1ULL << target0_qubit_index); // 00001
		basis[2] = basis0 + (1ULL << target1_qubit_index); // 00010
		basis[3] = basis0 + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 00011
		basis[4] = basis0 + (1ULL << target2_qubit_index); // 00100
		basis[5] = basis0 + (1ULL << target2_qubit_index) + (1ULL << target0_qubit_index); // 00101
		basis[6] = basis0 + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index); // 00110
		basis[7] = basis0 + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 00111
		basis[8] = basis0 + (1ULL << target3_qubit_index); // 01000
		basis[9] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target0_qubit_index); // 01001
		basis[10] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target1_qubit_index); // 01010
		basis[11] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 01011
		basis[12] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target2_qubit_index); // 01100
		basis[13] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target2_qubit_index) + (1ULL << target0_qubit_index); // 01101
		basis[14] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index); // 01110
		basis[15] = basis0 + (1ULL << target3_qubit_index) + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 01111
		basis[16] = basis0 + (1ULL << target4_qubit_index); // 10000
		basis[17] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target0_qubit_index); // 10001
		basis[18] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target1_qubit_index); // 10010
		basis[19] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 10011
		basis[20] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target2_qubit_index); // 10100
		basis[21] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target2_qubit_index) + (1ULL << target0_qubit_index); // 10101
		basis[22] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index); // 10110
		basis[23] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 10111
		basis[24] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target3_qubit_index); // 11000
		basis[25] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target3_qubit_index) + (1ULL << target0_qubit_index); // 11001
		basis[26] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target3_qubit_index) + (1ULL << target1_qubit_index); // 11010
		basis[27] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target3_qubit_index) + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 11011
		basis[28] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target3_qubit_index) + (1ULL << target2_qubit_index); // 11100
		basis[29] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target3_qubit_index) + (1ULL << target2_qubit_index) + (1ULL << target0_qubit_index); // 11101
		basis[30] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target3_qubit_index) + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index); // 11110
		basis[31] = basis0 + (1ULL << target4_qubit_index) + (1ULL << target3_qubit_index) + (1ULL << target2_qubit_index) + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index); // 11111

		for(y = 0 ; y < 32 ; ++y ){
			d_buffer[y]=make_cuDoubleComplex(0.0,0.0);
			for(x = 0 ; x < 32 ; ++x){
				d_buffer[y] = cuCadd(d_buffer[y], 
						cuCmul(matrix_const_gpu[y*32 + x], state_gpu[ basis[x] ]));
			}
		}
        	for(y = 0 ; y < 32; ++y){
			state_gpu[basis[y]] = d_buffer[y];
		}
	}
}

__host__ void penta_qubit_dense_matrix_gate_host(unsigned int target_qubit_index[5], CPPCTYPE matrix[1024], void* state, ITYPE dim) {
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int target0_qubit_index, target1_qubit_index, target2_qubit_index, target3_qubit_index, target4_qubit_index;
	checkCudaErrors(cudaMemcpyToSymbol(matrix_const_gpu, matrix, sizeof(GTYPE)*1024), __FILE__, __LINE__);
	ITYPE loop_dim = dim >> 5;

	/*
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
	printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
	printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
	printf("  Warp size:                                     %d\n", deviceProp.warpSize);
	printf("  Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
	 */	
	
	unsigned int block = loop_dim <= 512 ? loop_dim : 512;
	unsigned int grid = loop_dim / block;

	std::sort(target_qubit_index, target_qubit_index+5);

	target0_qubit_index=target_qubit_index[0];
	target1_qubit_index=target_qubit_index[1];
	target2_qubit_index=target_qubit_index[2];
	target3_qubit_index=target_qubit_index[3];
	target4_qubit_index=target_qubit_index[4];

	penta_qubit_dense_matrix_gate_gpu << <grid, block >> >(target4_qubit_index, target3_qubit_index, target2_qubit_index, target1_qubit_index, target0_qubit_index, state_gpu, dim);
	
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}




__global__ void quad_qubit_dense_matrix_gate_gpu(unsigned int target3_qubit_index, unsigned int target2_qubit_index, unsigned int target1_qubit_index, unsigned int target0_qubit_index, GTYPE *state_gpu, ITYPE dim){
	//ITYPE basis0;
	ITYPE basis[16];
	GTYPE d_buffer[16];
	ITYPE loop_dim = dim >> 4;
	ITYPE basis0 = blockIdx.x * blockDim.x + threadIdx.x;

	int x, y;

	if (basis0 < loop_dim){
        	//basis0 = j;
		// create base index
		basis0 = insert_zero_to_basis_index_device(basis0, target0_qubit_index );
		basis0 = insert_zero_to_basis_index_device(basis0, target1_qubit_index );
		basis0 = insert_zero_to_basis_index_device(basis0, target2_qubit_index );
		basis0 = insert_zero_to_basis_index_device(basis0, target3_qubit_index );
		
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
				//basis = basis0+(((x>>0)&1)<<target_qubit_index_const_gpu[0])+(((x>>1)&1)<<target_qubit_index_const_gpu[1])+(((x>>2)&1)<<target_qubit_index_const_gpu[2])+(((x>>3)&1)<<target_qubit_index_const_gpu[3]);
				d_buffer[y] = cuCadd(d_buffer[y], 
						cuCmul(matrix_const_gpu[y*16 + x], state_gpu[ basis[x] ]));
			}
		}
        	for(y = 0 ; y < 16 ; ++y){
			//basis = basis0+(((y>>0)&1)<<target_qubit_index_const_gpu[0])+(((y>>1)&1)<<target_qubit_index_const_gpu[1])+(((y>>2)&1)<<target_qubit_index_const_gpu[2])+(((y>>3)&1)<<target_qubit_index_const_gpu[3]);
			state_gpu[basis[y]] = d_buffer[y];
		}
	}
}

__host__ void quad_qubit_dense_matrix_gate_host(unsigned int target_qubit_index[4], CPPCTYPE matrix[256], void* state, ITYPE dim) {
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int target0_qubit_index, target1_qubit_index, target2_qubit_index, target3_qubit_index;
	checkCudaErrors(cudaMemcpyToSymbol(matrix_const_gpu, matrix, sizeof(GTYPE)*256), __FILE__, __LINE__);
	ITYPE loop_dim = dim >> 4;

	/*
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
	printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
	printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
	printf("  Warp size:                                     %d\n", deviceProp.warpSize);
	printf("  Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
	 */	
	
	unsigned int block = loop_dim <= 512 ? loop_dim : 512;
	unsigned int grid = loop_dim / block;

	std::sort(target_qubit_index, target_qubit_index+4);

	target0_qubit_index=target_qubit_index[0];
	target1_qubit_index=target_qubit_index[1];
	target2_qubit_index=target_qubit_index[2];
	target3_qubit_index=target_qubit_index[3];

	quad_qubit_dense_matrix_gate_gpu << <grid, block >> >(target3_qubit_index, target2_qubit_index, target1_qubit_index, target0_qubit_index, state_gpu, dim);
	
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}


// target qubit 1 > target qubit 2 > target qubit 3

__global__ void triple_qubit_dense_matrix_gate_gpu(unsigned int target1_qubit_index, unsigned int target2_qubit_index, unsigned int target3_qubit_index, 
		GTYPE *state_gpu, ITYPE dim){
	unsigned int left, mid, right;
	ITYPE basis[8];
	GTYPE d_buffer[8];
	ITYPE loop_dim = dim >> 3;
	ITYPE basis0 = blockIdx.x * blockDim.x + threadIdx.x;

	int x, y;
	left = target1_qubit_index;
	mid = target2_qubit_index;
	right = target3_qubit_index;
	
	if (basis0 < loop_dim){
		// create base index
		basis0 = insert_zero_to_basis_index_device(basis0, right );
		basis0 = insert_zero_to_basis_index_device(basis0, mid );
		basis0 = insert_zero_to_basis_index_device(basis0, left );
		
		basis[0] =  basis0; // 000
		basis[1] =  basis0 + (1ULL << right); // 001
		basis[2] = basis0 + (1ULL << mid); // 010
		basis[3] = basis0 + (1ULL << mid) + (1ULL << right); // 011
		basis[4] = basis0 + (1ULL << left); // 100
		basis[5] = basis0 + (1ULL << left) + (1ULL << right); // 101
		basis[6] = basis0 + (1ULL << left) + (1ULL << mid); // 110
		basis[7] = basis0 + (1ULL << left) + (1ULL << mid) + (1ULL << right); // 111

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

__host__ void triple_qubit_dense_matrix_gate_host(unsigned int target1_qubit_index, unsigned int target2_qubit_index, unsigned int target3_qubit_index, CPPCTYPE matrix[64], void* state, ITYPE dim) {
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;

	unsigned int left, mid, right, tmp;
	
	left = target1_qubit_index;
	mid = target2_qubit_index;
	right = target3_qubit_index;
	
	if(left > mid){ tmp=left; left=mid; mid=left;}
	if(mid > right){ tmp=right; right=mid; mid=tmp;}
	if(left > mid){ tmp=left; left=mid; mid=left;}
	
	checkCudaErrors(cudaMemcpyToSymbol(matrix_const_gpu, matrix, sizeof(GTYPE)*64), __FILE__, __LINE__);
	ITYPE loop_dim = dim >> 3;
	unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;
	
	triple_qubit_dense_matrix_gate_gpu << <grid, block >> >(target1_qubit_index, target2_qubit_index, target3_qubit_index, state_gpu, dim);
	
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

// target1 qubit index > target0 qubit index
__global__ void double_qubit_dense_matrix_gate_gpu_local_variable(unsigned int target1_qubit_index, unsigned int target0_qubit_index, GTYPE *state_gpu, ITYPE dim){
	// unsigned int left, right;
	ITYPE head, body, tail;
	ITYPE basis0, basis1, basis2, basis3;
	GTYPE d_buffer0, d_buffer1, d_buffer2, d_buffer3;
	ITYPE quad_dim = dim >> 2;
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;

	// target1 qubit index > target2 qubit index
	
	if (j < quad_dim){
		head = j >> (target1_qubit_index - 1);
		body = (j & ((1ULL << (target1_qubit_index - 1)) - 1)) >> target0_qubit_index; // (j % 2^(k-1)) >> i
		tail = j & ((1ULL << target0_qubit_index) - 1); // j%(2^i)

		basis0 =  (head << (target1_qubit_index + 1)) + (body << (target0_qubit_index + 1)) + tail;
		basis1 = basis0 + (1ULL << target0_qubit_index);
		basis2 = basis0 + (1ULL << target1_qubit_index);
		basis3 = basis0 + (1ULL << target1_qubit_index) + (1ULL << target0_qubit_index);

		d_buffer0 = make_cuDoubleComplex(0.0,0.0);
		d_buffer0 = cuCadd(d_buffer0, cuCmul(matrix_const_gpu[0], state_gpu[ basis0 ]));
		d_buffer0 = cuCadd(d_buffer0, cuCmul(matrix_const_gpu[1], state_gpu[ basis1 ]));
		d_buffer0 = cuCadd(d_buffer0, cuCmul(matrix_const_gpu[2], state_gpu[ basis2 ]));
		d_buffer0 = cuCadd(d_buffer0, cuCmul(matrix_const_gpu[3], state_gpu[ basis3 ]));
		d_buffer1 = make_cuDoubleComplex(0.0,0.0);
		d_buffer1 = cuCadd(d_buffer1, cuCmul(matrix_const_gpu[4], state_gpu[ basis0 ]));
		d_buffer1 = cuCadd(d_buffer1, cuCmul(matrix_const_gpu[5], state_gpu[ basis1 ]));
		d_buffer1 = cuCadd(d_buffer1, cuCmul(matrix_const_gpu[6], state_gpu[ basis2 ]));
		d_buffer1 = cuCadd(d_buffer1, cuCmul(matrix_const_gpu[7], state_gpu[ basis3 ]));
		d_buffer2 = make_cuDoubleComplex(0.0,0.0);
		d_buffer2 = cuCadd(d_buffer2, cuCmul(matrix_const_gpu[8], state_gpu[ basis0 ]));
		d_buffer2 = cuCadd(d_buffer2, cuCmul(matrix_const_gpu[9], state_gpu[ basis1 ]));
		d_buffer2 = cuCadd(d_buffer2, cuCmul(matrix_const_gpu[10], state_gpu[ basis2 ]));
		d_buffer2 = cuCadd(d_buffer2, cuCmul(matrix_const_gpu[11], state_gpu[ basis3 ]));
		d_buffer3 = make_cuDoubleComplex(0.0,0.0);
		d_buffer3 = cuCadd(d_buffer3, cuCmul(matrix_const_gpu[12], state_gpu[ basis0 ]));
		d_buffer3 = cuCadd(d_buffer3, cuCmul(matrix_const_gpu[13], state_gpu[ basis1 ]));
		d_buffer3 = cuCadd(d_buffer3, cuCmul(matrix_const_gpu[14], state_gpu[ basis2 ]));
		d_buffer3 = cuCadd(d_buffer3, cuCmul(matrix_const_gpu[15], state_gpu[ basis3 ]));
        	
		state_gpu[basis0] = d_buffer0;
		state_gpu[basis1] = d_buffer1;
		state_gpu[basis2] = d_buffer2;
		state_gpu[basis3] = d_buffer3;
	}
}


// target1 qubit index > target0 qubit index
__global__ void double_qubit_dense_matrix_gate_gpu(unsigned int target1_qubit_index, unsigned int target0_qubit_index, GTYPE *state_gpu, ITYPE dim){
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
		head = j >> (target1_qubit_index - 1);
		body = (j & ((1ULL << (target1_qubit_index - 1)) - 1)) >> target0_qubit_index; // (j % 2^(k-1)) >> i
		tail = j & ((1ULL << target0_qubit_index) - 1); // j%(2^i)

		basis0 =  (head << (target1_qubit_index + 1)) + (body << (target0_qubit_index + 1)) + tail;
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

__host__ void double_qubit_dense_matrix_gate_host(unsigned int target1_qubit_index, unsigned int target0_qubit_index, CPPCTYPE matrix[16], void* state, ITYPE dim) {
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int tmp;
	if(target1_qubit_index < target0_qubit_index){
		tmp=target1_qubit_index; target1_qubit_index=target0_qubit_index; target0_qubit_index=tmp;
	}

	checkCudaErrors(cudaMemcpyToSymbol(matrix_const_gpu, matrix, sizeof(GTYPE)*16), __FILE__, __LINE__);
	ITYPE quad_dim = dim >> 2;
	unsigned int block = quad_dim <= 1024 ? quad_dim : 1024;
	unsigned int grid = quad_dim / block;
	
	double_qubit_dense_matrix_gate_gpu << <grid, block >> >(target1_qubit_index, target0_qubit_index, state_gpu, dim);
	
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__host__ void double_qubit_dense_matrix_gate_local_variable_host(unsigned int target1_qubit_index, unsigned int target0_qubit_index, CPPCTYPE matrix[16], void* state, ITYPE dim) {
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int tmp;
	if(target1_qubit_index < target0_qubit_index){
		tmp=target1_qubit_index; target1_qubit_index=target0_qubit_index; target0_qubit_index=tmp;
	}

	checkCudaErrors(cudaMemcpyToSymbol(matrix_const_gpu, matrix, sizeof(GTYPE)*16), __FILE__, __LINE__);
	ITYPE quad_dim = dim >> 2;
	unsigned int block = quad_dim <= 1024 ? quad_dim : 1024;
	unsigned int grid = quad_dim / block;
	double_qubit_dense_matrix_gate_gpu_local_variable<< <grid, block >> >(target1_qubit_index, target0_qubit_index, state_gpu, dim);
	
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
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

__host__ void multi_qubit_Pauli_gate_Z_mask_host(ITYPE phase_flip_mask, void* state, ITYPE dim){
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	
    multi_qubit_Pauli_gate_Z_mask_gpu << <grid, block >> >(phase_flip_mask, state_gpu, dim);		
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
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
		//unsigned int sign_0 = popcount64(basis_0 & phase_flip_mask)&1;
		//unsigned int sign_1 = popcount64(basis_1 & phase_flip_mask)&1;
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

__host__ void multi_qubit_Pauli_gate_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, void* state, ITYPE dim){
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	
    multi_qubit_Pauli_gate_XZ_mask_gpu<< <grid, block >> >(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_gpu, dim);
	
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

/*
const CPPCTYPE PHASE_90ROT[4] = {1., 1.i, -1, -1.i};
const CPPCTYPE PHASE_M90ROT[4] = { 1., -1.i, -1, 1.i };
*/

__device__ void multi_qubit_Pauli_rotation_gate_XZ_mask_device(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	// loop varaibles
	ITYPE loop_dim = dim>>1;

	// coefs
	double cosval = cos(angle);
	double sinval = sin(angle);
	//GTYPE PHASE_90ROT[4] = {make_cuDoubleComplex(1.0,0.0), make_cuDoubleComplex(0.0,1.0), make_cuDoubleComplex(-1.0,0.0), make_cuDoubleComplex(0.0,-1.0)};
	GTYPE PHASE_M90ROT[4] = { make_cuDoubleComplex(1.0,0.0), make_cuDoubleComplex(0.0,-1), make_cuDoubleComplex(-1,0.0), make_cuDoubleComplex(0.0,1)};
	if(state_index<loop_dim){
		// create base index
		ITYPE basis_0 = insert_zero_to_basis_index_device(state_index, pivot_qubit_index);
		// gather index
		ITYPE basis_1 = basis_0 ^ bit_flip_mask;
		// determine parity
		//unsigned int bit_parity_0 = popcount64(basis_0 & phase_flip_mask)&1;
		//unsigned int bit_parity_1 = popcount64(basis_1 & phase_flip_mask)&1;
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

__host__ void multi_qubit_Pauli_rotation_gate_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, void* state, ITYPE dim){
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	
	multi_qubit_Pauli_rotation_gate_XZ_mask_gpu<< <grid, block >> >(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle, state_gpu, dim);
    
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__device__ void multi_qubit_Pauli_rotation_gate_Z_mask_device(ITYPE phase_flip_mask, double angle, GTYPE* state_gpu, ITYPE dim){
	// loop variables
	const ITYPE loop_dim = dim;
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	// coefs
	const double cosval = cos(angle);
	const double sinval = sin(angle);

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

__host__ void multi_qubit_Pauli_rotation_gate_Z_mask_host(ITYPE phase_flip_mask, double angle, void* state, ITYPE dim){
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	
	multi_qubit_Pauli_rotation_gate_Z_mask_gpu<< <grid, block >> >(phase_flip_mask, angle, state_gpu, dim);
	
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__host__ void multi_qubit_Pauli_gate_partial_list_host(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, void* state, ITYPE dim){
	// create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_partial_list_gsim(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if(bit_flip_mask == 0){
		multi_qubit_Pauli_gate_Z_mask_host(phase_flip_mask, state, dim);
	}else{
		multi_qubit_Pauli_gate_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim);
	}
}

__host__ void multi_qubit_Pauli_gate_whole_list_host(const UINT* Pauli_operator_type_list, UINT qubit_count, void* state, ITYPE dim){
	 // create pauli mask and call function
	 ITYPE bit_flip_mask = 0;
	 ITYPE phase_flip_mask = 0;
	 UINT global_phase_90rot_count = 0;
	 UINT pivot_qubit_index = 0;
	 get_Pauli_masks_whole_list_gsim(Pauli_operator_type_list, qubit_count,
		 &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	 if(bit_flip_mask == 0){
		 multi_qubit_Pauli_gate_Z_mask_host(phase_flip_mask, state, dim);
	 }else{
		 multi_qubit_Pauli_gate_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim);
	 }
 } 

__host__ void multi_qubit_Pauli_rotation_gate_partial_list_host(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, double angle, void* state, ITYPE dim){
	 // create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_partial_list_gsim(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if(bit_flip_mask == 0){
		multi_qubit_Pauli_rotation_gate_Z_mask_host(phase_flip_mask, angle, state, dim);
	}else{
		multi_qubit_Pauli_rotation_gate_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index,angle, state, dim);
	}
}
 
__host__ void multi_qubit_Pauli_rotation_gate_whole_list_host(const UINT* Pauli_operator_type_list, UINT qubit_count, double angle, void* state, ITYPE dim){
	// create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_whole_list_gsim(Pauli_operator_type_list, qubit_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if(bit_flip_mask == 0){
		multi_qubit_Pauli_rotation_gate_Z_mask_host(phase_flip_mask, angle, state, dim);
	}else{
		multi_qubit_Pauli_rotation_gate_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle, state, dim);
	}
}


// __constant__ GTYPE matrix_const_gpu[1024];
// __constant__ ITYPE matrix_mask_list_gpu[1024];
// __constant__ UINT sorted_insert_index_list_gpu[10];
__global__ void multi_qubit_dense_matrix_gate_gpu(UINT target_qubit_index_count, GTYPE* matrix_gpu, 
	GTYPE* d_buffer, ITYPE* matrix_mask_list, UINT* sorted_insert_index_list, GTYPE* state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE x, y;
	// matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    
    // loop variables
	const ITYPE loop_dim = dim >> target_qubit_index_count;
	if(state_index<loop_dim){
		// create base index
        ITYPE basis_0 = state_index;
        for(int cursor=0; cursor < target_qubit_index_count ; cursor++){
			UINT insert_index = sorted_insert_index_list[cursor];
			basis_0 = insert_zero_to_basis_index_device(basis_0, insert_index );
		}
		// compute matrix-vector multiply
		for(y = 0 ; y < matrix_dim ; ++y ){
			d_buffer[y]=make_cuDoubleComplex(0.0,0.0);
			for(x = 0 ; x < matrix_dim ; ++x){
				d_buffer[y] = cuCadd(d_buffer[y], 
					cuCmul(matrix_gpu[y*matrix_dim + x], state_gpu[ basis_0 ^ matrix_mask_list[x] ]));
			}
		}
		// set result
        for(y = 0 ; y < matrix_dim ; ++y){
			state_gpu[basis_0 ^ matrix_mask_list[y]] = d_buffer[y];
        }
    }
}

// target_qubit_count <= 5 or < 5
__global__ void multi_qubit_dense_matrix_gate_const_gpu(UINT target_qubit_index_count, GTYPE* state_gpu, ITYPE dim){
    GTYPE d_buffer[32];
    ITYPE state_index  = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE x, y;
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    
	if(state_index < (dim >> target_qubit_index_count)){
        // create base index
        ITYPE basis_0 = state_index;
        unsigned int insert_index;
        for(int cursor=0; cursor < target_qubit_index_count ; cursor++){
			insert_index = sorted_insert_index_list_gpu[cursor];
			basis_0 = insert_zero_to_basis_index_device(basis_0, insert_index );
		}

		// compute matrix-vector multiply
		for(y = 0 ; y < matrix_dim ; ++y ){
			d_buffer[y]=make_cuDoubleComplex(0.0,0.0);
			for(x = 0 ; x < matrix_dim ; ++x){
				d_buffer[y] = cuCadd(d_buffer[y], 
                        cuCmul(matrix_const_gpu[y*matrix_dim + x], state_gpu[ basis_0 ^ matrix_mask_list_gpu[x] ])
                    );
			}
		}

		// set result
        for(y = 0 ; y < matrix_dim ; ++y){
			state_gpu[basis_0 ^ matrix_mask_list_gpu[y]] = d_buffer[y];
        }
    }
}

// target_qubit_count <= 10, num_elem: 1024
__global__ void multi_qubit_dense_matrix_gate_const_gpu(UINT target_qubit_index_count, GTYPE* matrix_gpu, GTYPE* state_gpu, ITYPE dim){
    GTYPE d_buffer[1024];
    ITYPE state_index  = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE x, y;
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    
	if(state_index < (dim >> target_qubit_index_count)){
        // create base index
        ITYPE basis_0 = state_index;
        unsigned int insert_index;
        for(int cursor=0; cursor < target_qubit_index_count ; cursor++){
			insert_index = sorted_insert_index_list_gpu[cursor];
			basis_0 = insert_zero_to_basis_index_device(basis_0, insert_index );
		}

		// compute matrix-vector multiply
		for(y = 0 ; y < matrix_dim ; ++y ){
			d_buffer[y]=make_cuDoubleComplex(0.0,0.0);
			for(x = 0 ; x < matrix_dim ; ++x){
				d_buffer[y] = cuCadd(d_buffer[y], 
                        cuCmul(matrix_gpu[y*matrix_dim + x], state_gpu[ basis_0 ^ matrix_mask_list_gpu[x] ])
                    );
			}
		}

		// set result
        for(y = 0 ; y < matrix_dim ; ++y){
			state_gpu[basis_0 ^ matrix_mask_list_gpu[y]] = d_buffer[y];
        }
    }
}

__host__ void multi_qubit_dense_matrix_gate_host(UINT* target_qubit_index_list, UINT target_qubit_index_count, const CPPCTYPE* matrix, void* state, ITYPE dim){
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;

	// matrix dim, mask, buffer
    ITYPE matrix_dim = 1ULL << target_qubit_index_count;
	ITYPE* h_matrix_mask_list = create_matrix_mask_list_gsim(target_qubit_index_list, target_qubit_index_count);

    // insert index
    UINT* h_sorted_insert_index_list = create_sorted_ui_list_gsim(target_qubit_index_list, target_qubit_index_count);

    // loop variables
	ITYPE loop_dim = dim >> target_qubit_index_count;
	
	GTYPE* d_buffer;
    GTYPE* matrix_gpu;
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
		    checkCudaErrors(cudaMemcpyToSymbol(matrix_const_gpu, matrix, sizeof(GTYPE)*matrix_dim*matrix_dim), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbol(sorted_insert_index_list_gpu, h_sorted_insert_index_list, sizeof(UINT)*target_qubit_index_count), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbol(matrix_mask_list_gpu, h_matrix_mask_list, sizeof(ITYPE)*matrix_dim), __FILE__, __LINE__);
		    multi_qubit_dense_matrix_gate_const_gpu << <grid, block >> >(target_qubit_index_count, state_gpu, dim);
        }else{
		    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&matrix_gpu), matrix_dim *matrix_dim * sizeof(GTYPE) ), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpy(matrix_gpu, matrix, matrix_dim *matrix_dim * sizeof(GTYPE), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbol(sorted_insert_index_list_gpu, h_sorted_insert_index_list, sizeof(UINT)*target_qubit_index_count), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbol(matrix_mask_list_gpu, h_matrix_mask_list, sizeof(ITYPE)*matrix_dim), __FILE__, __LINE__);
		    multi_qubit_dense_matrix_gate_const_gpu << <grid, block >> >(target_qubit_index_count, matrix_gpu, state_gpu, dim);
        }
	}else{
        assert(0);
        // printf("this function may be invalid.\n");
	    ITYPE* d_matrix_mask_list;
    	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_matrix_mask_list), matrix_dim * sizeof(ITYPE) ), __FILE__, __LINE__);
	    checkCudaErrors(cudaMemcpy(d_matrix_mask_list, h_matrix_mask_list, matrix_dim * sizeof(ITYPE), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	    UINT* d_sorted_insert_index_list;
	    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_sorted_insert_index_list), matrix_dim * sizeof(ITYPE) ), __FILE__, __LINE__);
	    checkCudaErrors(cudaMemcpy(d_sorted_insert_index_list, h_sorted_insert_index_list, matrix_dim * sizeof(ITYPE), cudaMemcpyHostToDevice), __FILE__, __LINE__);
        
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer), matrix_dim * matrix_dim * sizeof(GTYPE) ), __FILE__, __LINE__);
        checkCudaErrors(cudaMemset(d_buffer, 0, matrix_dim * matrix_dim * sizeof(GTYPE)), __FILE__, __LINE__);
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&matrix_gpu), matrix_dim *matrix_dim * sizeof(GTYPE) ), __FILE__, __LINE__);
		checkCudaErrors(cudaMemcpy(matrix_gpu, matrix, matrix_dim *matrix_dim * sizeof(GTYPE), cudaMemcpyHostToDevice), __FILE__, __LINE__);
        multi_qubit_dense_matrix_gate_gpu << <grid, block >> >(target_qubit_index_count, matrix_gpu, d_buffer, d_matrix_mask_list, d_sorted_insert_index_list, state_gpu, dim);
	    cudaFree(d_sorted_insert_index_list);
	    cudaFree(d_matrix_mask_list);
        cudaFree(d_buffer);
	}
    
	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
    
    // Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus, __FILE__, __LINE__);
    
    if(target_qubit_index_count>5) cudaFree(matrix_gpu);
    free((UINT*)h_sorted_insert_index_list);
	free((ITYPE*)h_matrix_mask_list);
	
    state = reinterpret_cast<void*>(state_gpu);
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

// this function is invalid
__global__ void single_qubit_control_multi_qubit_dense_matrix_gate_gpu(UINT control_qubit_index, UINT control_value, UINT target_qubit_index_count, const GTYPE* matrix, ITYPE* matrix_mask_list, UINT* sorted_insert_index_list, GTYPE* state, ITYPE dim) {

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
            UINT insert_index = sorted_insert_index_list[cursor];
            basis_0 = insert_zero_to_basis_index_device(basis_0, insert_index );
        }

        // flip control
        basis_0 ^= control_mask;

       // compute matrix mul
       for(ITYPE y = 0 ; y < matrix_dim ; ++y ){
            d_buffer[y]=make_cuDoubleComplex(0.0, 0.0);
            for(ITYPE x = 0 ; x < matrix_dim ; ++x){
                d_buffer[y] = cuCadd( d_buffer[y], cuCmul( matrix[y*matrix_dim + x], state[ basis_0 ^ matrix_mask_list[x] ]));
			}
        }

        // set result
        for(ITYPE y = 0 ; y < matrix_dim ; ++y){
            state[basis_0 ^ matrix_mask_list[y]] = d_buffer[y];
        }
    }
}

__host__ void single_qubit_control_multi_qubit_dense_matrix_gate_host(UINT control_qubit_index, UINT control_value, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CPPCTYPE* matrix, void* state, ITYPE dim) {
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
    // matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    ITYPE* matrix_mask_list = create_matrix_mask_list_gsim(target_qubit_index_list, target_qubit_index_count);

    // insert list
    const UINT insert_index_count = target_qubit_index_count + 1;
    UINT* sorted_insert_index_list = create_sorted_ui_list_value_gsim(target_qubit_index_list, target_qubit_index_count ,control_qubit_index);

    GTYPE *d_matrix, *d_matrix_mask_list, *d_sorted_insert_index_list;

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
		    checkCudaErrors(cudaMemcpyToSymbol(matrix_const_gpu, matrix, sizeof(GTYPE)*matrix_dim*matrix_dim), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbol(d_matrix_mask_list, matrix, sizeof(GTYPE)*matrix_dim), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbol(d_sorted_insert_index_list, matrix, sizeof(GTYPE)*matrix_dim), __FILE__, __LINE__);

            single_qubit_control_multi_qubit_dense_matrix_gate_const_gpu<<< grid, block >>> (control_qubit_index, control_value, target_qubit_index_count, state_gpu, dim);
        }else{
		    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_matrix), matrix_dim *matrix_dim * sizeof(GTYPE) ), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpy(d_matrix, matrix, matrix_dim *matrix_dim * sizeof(GTYPE), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbol(d_matrix_mask_list, matrix, sizeof(GTYPE)*matrix_dim), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbol(d_sorted_insert_index_list, matrix, sizeof(GTYPE)*matrix_dim), __FILE__, __LINE__);
		    
            single_qubit_control_multi_qubit_dense_matrix_gate_const_gpu<<< grid, block >>> (control_qubit_index, control_value, target_qubit_index_count, d_matrix, state_gpu, dim);
        }
	}else{
        // printf("this function may be invalid.\n");
        // target_qubit_index > 10
        assert(0);
	}

	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
    
    // Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus, __FILE__, __LINE__);
    
	cudaFree(d_sorted_insert_index_list);
	cudaFree(d_matrix_mask_list);
    // if(target_qubit_index_count>10) cudaFree(d_buffer);
    if(target_qubit_index_count>5) cudaFree(d_matrix);
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

__host__ void multi_qubit_control_multi_qubit_dense_matrix_gate_host(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, const UINT* target_qubit_index_list, UINT target_qubit_index_count, const CPPCTYPE* matrix, void* state, ITYPE dim) {
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

    GTYPE *d_matrix, *d_matrix_mask_list, *d_sorted_insert_index_list;

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
		    checkCudaErrors(cudaMemcpyToSymbol(matrix_const_gpu, matrix, sizeof(GTYPE)*matrix_dim*matrix_dim), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbol(d_matrix_mask_list, matrix, sizeof(GTYPE)*matrix_dim), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbol(d_sorted_insert_index_list, matrix, sizeof(GTYPE)*matrix_dim), __FILE__, __LINE__);

            multi_qubit_control_multi_qubit_dense_matrix_gate_const_gpu<<< grid, block >>> (control_mask, target_qubit_index_count, control_qubit_index_count, state_gpu, dim);
        }else{
		    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_matrix), matrix_dim *matrix_dim * sizeof(GTYPE) ), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpy(d_matrix, matrix, matrix_dim *matrix_dim * sizeof(GTYPE), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbol(d_matrix_mask_list, matrix, sizeof(GTYPE)*matrix_dim), __FILE__, __LINE__);
		    checkCudaErrors(cudaMemcpyToSymbol(d_sorted_insert_index_list, matrix, sizeof(GTYPE)*matrix_dim), __FILE__, __LINE__);
		    
            multi_qubit_control_multi_qubit_dense_matrix_gate_const_gpu<<< grid, block >>> (control_mask, target_qubit_index_count, control_qubit_index_count, d_matrix, state_gpu, dim);
        }
	}else{
        // printf("this function may be invalid.\n");
        // target_qubit_index > 10
        assert(0);
	}

	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
    
    // Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus, __FILE__, __LINE__);
    
	cudaFree(d_sorted_insert_index_list);
	cudaFree(d_matrix_mask_list);
    // if(target_qubit_index_count>10) cudaFree(d_buffer);
    if(target_qubit_index_count>5) cudaFree(d_matrix);
    free(sorted_insert_index_list);
    free(matrix_mask_list);
 
    state = reinterpret_cast<void*>(state_gpu);
}

