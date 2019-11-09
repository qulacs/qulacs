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

__global__ void H_gate_gpu(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim) {
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE basis0, basis1;
	GTYPE tmp;
	double inv_sqrt=1.0/sqrt(2.0);

	if (j < (dim >> 1)){
		//basis0 = ((j & ~((ONE<< i)-1)) << 1) + (j & ((ONE<< i)-1));
		//basis1 = basis0 + (ONE<< i);
		basis0 = (j >> target_qubit_index);
		basis0 = basis0 << (target_qubit_index + 1);
		basis0 += j & ((1ULL << target_qubit_index) - 1);
		basis1 = basis0 ^ (1ULL << target_qubit_index);

		tmp = state_gpu[basis0];
		state_gpu[basis0] = cuCadd(tmp, state_gpu[basis1]);
		state_gpu[basis1] = cuCadd(tmp, make_cuDoubleComplex(-1*state_gpu[basis1].x, -1*state_gpu[basis1].y));
		state_gpu[basis0] = make_cuDoubleComplex(state_gpu[basis0].x * inv_sqrt, state_gpu[basis0].y * inv_sqrt);
		state_gpu[basis1] = make_cuDoubleComplex(state_gpu[basis1].x * inv_sqrt, state_gpu[basis1].y * inv_sqrt);
	}
}

__host__ void H_gate_host(unsigned int target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice((int)device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	cudaError cudaStatus;
	ITYPE half_dim = dim >> 1;
	unsigned int block = half_dim <= 1024 ? half_dim : 1024;
	unsigned int grid = half_dim / block;

	H_gate_gpu << <grid, block, 0, *cuda_stream >> > (target_qubit_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__global__ void X_gate_gpu(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim) {
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE basis0, basis1;
	GTYPE tmp;

	if (j < (dim>>1)){
		//basis0 = ((j & ~((ONE<< i)-1)) << 1) + (j & ((ONE<< i)-1));
		//basis1 = basis0 + (ONE<< i);
		basis0 = (j >> target_qubit_index);
		basis0 = basis0 << (target_qubit_index + 1);
		basis0 += j & ((1ULL << target_qubit_index) - 1);
		basis1 = basis0 ^ (1ULL << target_qubit_index);

		tmp = state_gpu[basis0];
		state_gpu[basis0] = state_gpu[basis1];
		state_gpu[basis1] = tmp;
    }
}

__host__ void X_gate_host(unsigned int target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	cudaError cudaStatus;
	ITYPE half_dim = dim >> 1;
	unsigned int block = half_dim <= 1024 ? half_dim : 1024;
	unsigned int grid = half_dim / block;

	X_gate_gpu << <grid, block, 0, *cuda_stream >> > (target_qubit_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__global__ void Y_gate_gpu(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE dim) {
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE basis0, basis1;
	GTYPE tmp;

	if (j < (dim>>1)){
		basis0 = (j >> target_qubit_index);
		basis0 = basis0 << (target_qubit_index + 1);
		basis0 += j & ((1ULL << target_qubit_index) - 1);
		basis1 = basis0 ^ (1ULL << target_qubit_index);

		tmp = state_gpu[basis0];
		state_gpu[basis0] = make_cuDoubleComplex(cuCimag(state_gpu[basis1]), -cuCreal(state_gpu[basis1]));
		state_gpu[basis1] = make_cuDoubleComplex(-cuCimag(tmp), cuCreal(tmp));
	}
}

__host__ void Y_gate_host(unsigned int target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	cudaError cudaStatus;
	ITYPE half_dim = dim >> 1;
	unsigned int block = half_dim <= 1024 ? half_dim : 1024;
	unsigned int grid = half_dim / block;

	Y_gate_gpu << <grid, block, 0, *cuda_stream >> > (target_qubit_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__global__ void Z_gate_gpu(unsigned int target_qubit_index, GTYPE *state_gpu, ITYPE DIM) {
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE basis0, basis1;
	if (j < (DIM>>1)){
		basis0 = insert_zero_to_basis_index_device(j, target_qubit_index);
		basis1 = basis0^(1ULL<<target_qubit_index);
		state_gpu[basis1] = make_cuDoubleComplex(-cuCreal(state_gpu[basis1]), -cuCimag(state_gpu[basis1]));
	}
}

__host__ void Z_gate_host(unsigned int target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	cudaError cudaStatus;
	ITYPE half_dim = dim >> 1;
	unsigned int block = half_dim <= 1024 ? half_dim : 1024;
	unsigned int grid = half_dim / block;

	Z_gate_gpu << <grid, block, 0, *cuda_stream >> > (target_qubit_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__global__ void CZ_gate_gpu(unsigned int large_index, unsigned int small_index, GTYPE *state_gpu, ITYPE DIM) {
    ITYPE head, body, tail;
    ITYPE basis11;
    ITYPE quarter_DIM = DIM >> 2;
    ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < quarter_DIM){
        head = j >> (large_index - 1);
        body = (j & ((1ULL << (large_index- 1)) - 1)) >> small_index; // (j % 2^(large-1)) >> small
        tail = j & ((1ULL << small_index) - 1); // j%(2^small)
        
        basis11 = (head << (large_index + 1)) + (body << (small_index + 1)) + (1ULL << large_index) + (1ULL << small_index) + tail;
        
        state_gpu[basis11] = make_cuDoubleComplex(-cuCreal(state_gpu[basis11]), -cuCimag(state_gpu[basis11]));
	}
}

__host__ void CZ_gate_host(unsigned int control_qubit_index, unsigned int target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	cudaError cudaStatus;
	ITYPE quad_dim = dim >> 2;
	ITYPE large_index, small_index;

	if (control_qubit_index > target_qubit_index) {
		large_index = control_qubit_index;
		small_index = target_qubit_index;
	}
	else {
		large_index = target_qubit_index;
		small_index = control_qubit_index;
	}

	unsigned int block = quad_dim <= 1024 ? quad_dim : 1024;
	unsigned int grid = quad_dim / block;

	CZ_gate_gpu << <grid, block, 0, *cuda_stream >> > (large_index, small_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__global__ void CNOT_gate_gpu(unsigned int control_qubit_index, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim)
{
	unsigned int left, right;
	ITYPE head, body, tail;
	ITYPE basis10, basis11;
	GTYPE tmp_psi;
	ITYPE quarter_dim = dim >>2 ;
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;

	if (target_qubit_index > control_qubit_index){
		left = target_qubit_index;
		right = control_qubit_index;
	}
	else {
		left = control_qubit_index;
		right = target_qubit_index;
	}

	if (j < quarter_dim){
		head = j >> (left - 1);
		body = (j & ((1ULL << (left - 1)) - 1)) >> right; // (j % 2^(k-1)) >> i
		tail = j & ((1ULL << right) - 1); // j%(2^i)

		// ONE<<control
		basis10 = (head << (left + 1)) + (body << (right + 1)) + (1ULL << control_qubit_index) + tail;
		// ONE<<target, ONE<<control
		basis11 = (head << (left + 1)) + (body << (right + 1)) + (1ULL << target_qubit_index) + (1ULL << control_qubit_index) + tail;

		tmp_psi = psi_gpu[basis11];
		psi_gpu[basis11] = psi_gpu[basis10];
		psi_gpu[basis10] = tmp_psi;
	}
}

__host__ void CNOT_gate_host(unsigned int control_qubit_index, unsigned int target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	cudaError cudaStatus;
	ITYPE quad_dim = dim >> 2;
	unsigned int block = quad_dim <= 1024 ? quad_dim : 1024;
	unsigned int grid = quad_dim / block;

	CNOT_gate_gpu << <grid, block, 0, *cuda_stream >> > (control_qubit_index, target_qubit_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__global__ void SWAP_gate_gpu(unsigned int target_qubit_index0, unsigned int target_qubit_index1, GTYPE *state_gpu, ITYPE dim) {
	ITYPE head, body, tail;
	ITYPE basis01, basis10;
	GTYPE tmp;
	ITYPE quarter_dim = dim >> 2;
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < quarter_dim){
		head = j >> (target_qubit_index1 - 1);
		body = (j & ((1ULL << (target_qubit_index1 - 1)) - 1)) >> target_qubit_index0; // (j % 2^(k-1)) >> i
		tail = j & ((1ULL << target_qubit_index0) - 1); // j%(2^i)

		basis01 = (head << (target_qubit_index1 + 1)) + (body << (target_qubit_index0 + 1)) + (1ULL << target_qubit_index0) + tail;
		basis10 = (head << (target_qubit_index1 + 1)) + (body << (target_qubit_index0 + 1)) + (1ULL << target_qubit_index1) + tail;

		tmp = state_gpu[basis01];
		state_gpu[basis01] = state_gpu[basis10];
		state_gpu[basis10] = tmp;
	}
}

__host__ void SWAP_gate_host(unsigned int target_qubit_index0, unsigned int target_qubit_index1, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	cudaError cudaStatus;
	unsigned int large_index, small_index;
	ITYPE quad_dim = dim >> 2;
	unsigned int block = quad_dim <= 1024 ? quad_dim : 1024;
	unsigned int grid = quad_dim / block;

	if (target_qubit_index1 > target_qubit_index0) {
		large_index = target_qubit_index1;
		small_index = target_qubit_index0;
	}
	else {
		large_index = target_qubit_index0;
		small_index = target_qubit_index1;
	}

	SWAP_gate_gpu << <grid, block, 0, *cuda_stream >> > (small_index, large_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__global__ void P0_gate_gpu(UINT target_qubit_index, GTYPE *state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
    const ITYPE loop_dim = dim>>1;
    ITYPE mask = (1ULL << target_qubit_index);
    
    if(state_index<loop_dim){
        ITYPE tmp_index = insert_zero_to_basis_index_device(state_index, target_qubit_index) ^ mask;
        state_gpu[tmp_index] = make_cuDoubleComplex(0.0, 0.0);
    }
}

__host__ void P0_gate_host(UINT target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	cudaError cudaStatus;
	const ITYPE loop_dim = dim >> 1;

	unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;

	P0_gate_gpu << <grid, block, 0, *cuda_stream >> > (target_qubit_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__global__ void P1_gate_gpu(UINT target_qubit_index, GTYPE *state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
    const ITYPE loop_dim = dim>>1;
    
    if(state_index<loop_dim){
        ITYPE tmp_index = insert_zero_to_basis_index_device(state_index, target_qubit_index);
        state_gpu[tmp_index] = make_cuDoubleComplex(0.0, 0.0);
    }
}

__host__ void P1_gate_host(UINT target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	cudaError cudaStatus;
	const ITYPE loop_dim = dim >> 1;

	unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;

	P1_gate_gpu << <grid, block, 0, *cuda_stream >> > (target_qubit_index, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__global__ void normalize_gpu(const double normalize_factor, GTYPE *state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
    const ITYPE loop_dim = dim;
    
    if(state_index<loop_dim){
        state_gpu[state_index] = make_cuDoubleComplex(
                normalize_factor * cuCreal(state_gpu[state_index]),
                normalize_factor * cuCimag(state_gpu[state_index])
                );
    }
}

__host__ void normalize_host(double squared_norm, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	cudaError cudaStatus;
	const ITYPE loop_dim = dim;
	const double normalize_factor = sqrt(1./squared_norm);
	//const double normalize_factor = 1. / norm;

	unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;

	normalize_gpu << <grid, block, 0, *cuda_stream >> > (normalize_factor, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__host__ void RX_gate_host(UINT target_qubit_index, double angle, void* state, ITYPE dim, void* stream, unsigned int device_number){
    single_qubit_Pauli_rotation_gate_host(target_qubit_index, 1, angle, state, dim, stream, device_number);
}

__host__ void RY_gate_host(UINT target_qubit_index, double angle, void* state, ITYPE dim, void* stream, unsigned int device_number){
    single_qubit_Pauli_rotation_gate_host(target_qubit_index, 2, angle, state, dim, stream, device_number);
}

__host__ void RZ_gate_host(UINT target_qubit_index, double angle, void* state, ITYPE dim, void* stream, unsigned int device_number){
    single_qubit_Pauli_rotation_gate_host(target_qubit_index, 3, angle, state, dim, stream, device_number);
}

// [[1,0],[0,i]]
__host__ void S_gate_host(UINT target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	CPPCTYPE diagonal_matrix[2];
	diagonal_matrix[0] = CPPCTYPE(1.0, 0.0);
	diagonal_matrix[1] = CPPCTYPE(0.0, 1.0);
	single_qubit_diagonal_matrix_gate_host(target_qubit_index, diagonal_matrix, state, dim, stream, device_number);
}

// [[1,0],[0,-i]]
__host__ void Sdag_gate_host(UINT target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	CPPCTYPE diagonal_matrix[2];
	diagonal_matrix[0] = CPPCTYPE(1.0, 0.0);
	diagonal_matrix[1] = CPPCTYPE(0.0, -1.0);
	single_qubit_diagonal_matrix_gate_host(target_qubit_index, diagonal_matrix, state, dim, stream, device_number);
}

// [[1,0],[0,exp(i*pi/4)]] , (1+i)/sprt(2)
__host__ void T_gate_host(UINT target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	CPPCTYPE diagonal_matrix[2];
	diagonal_matrix[0] = CPPCTYPE(1.0, 0.0);
	diagonal_matrix[1] = CPPCTYPE(1.0 / sqrt(2), 1.0 / sqrt(2));
	single_qubit_diagonal_matrix_gate_host(target_qubit_index, diagonal_matrix, state, dim, stream, device_number);
}

// [[1,0],[0,-exp(i*pi/4)]], (1-i)/sqrt(2)
__host__ void Tdag_gate_host(UINT target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	CPPCTYPE diagonal_matrix[2];
	diagonal_matrix[0] = CPPCTYPE(1.0, 0.0);
	diagonal_matrix[1] = CPPCTYPE(1.0 / sqrt(2), -1.0 / sqrt(2));
	single_qubit_diagonal_matrix_gate_host(target_qubit_index, diagonal_matrix, state, dim, stream, device_number);
}

__host__ void sqrtX_gate_host(UINT target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	CPPCTYPE SQRT_X_GATE_MATRIX[4] = {
		std::complex<double>(0.5, 0.5), std::complex<double>(0.5, -0.5),
		std::complex<double>(0.5, -0.5), std::complex<double>(0.5, 0.5)
	};
	single_qubit_dense_matrix_gate_host(target_qubit_index, SQRT_X_GATE_MATRIX, state, dim, stream, device_number);
}

__host__ void sqrtXdag_gate_host(UINT target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	CPPCTYPE SQRT_X_DAG_GATE_MATRIX[4] =
	{
		std::complex<double>(0.5, -0.5), std::complex<double>(0.5, 0.5),
		std::complex<double>(0.5, 0.5), std::complex<double>(0.5, -0.5)
	};
	single_qubit_dense_matrix_gate_host(target_qubit_index, SQRT_X_DAG_GATE_MATRIX, state, dim, stream, device_number);
}

__host__ void sqrtY_gate_host(UINT target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	CPPCTYPE SQRT_Y_GATE_MATRIX[4] =
	{
		std::complex<double>(0.5, 0.5), std::complex<double>(-0.5, -0.5),
		std::complex<double>(0.5, 0.5), std::complex<double>(0.5, 0.5)
	};
	single_qubit_dense_matrix_gate_host(target_qubit_index, SQRT_Y_GATE_MATRIX, state, dim, stream, device_number);
}

__host__ void sqrtYdag_gate_host(UINT target_qubit_index, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	CPPCTYPE SQRT_Y_DAG_GATE_MATRIX[4] =
	{
		std::complex<double>(0.5, -0.5), std::complex<double>(0.5, -0.5),
		std::complex<double>(-0.5, 0.5), std::complex<double>(0.5, -0.5)
	};
	single_qubit_dense_matrix_gate_host(target_qubit_index, SQRT_Y_DAG_GATE_MATRIX, state, dim, stream, device_number);
}
