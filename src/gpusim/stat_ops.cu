#include "device_launch_parameters.h"
#include <cuda_runtime.h>
//#include <cuda.h>

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <complex>
//#include <sys/time.h>

#include <limits.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include "util_type.h"
#include "util_func.h"
#include "util.cuh"
#include "update_ops_cuda.h"
#include "stat_ops.h"

__constant__ GTYPE matrix_const_gpu[4];
__constant__ unsigned int num_pauli_op_gpu[4];
__constant__ ITYPE bit_mask_gpu[4];
__constant__ GTYPE PHASE_90ROT_gpu[4];

__device__ double atomicAdd_double(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

inline __device__ double __shfl_down_double(double var, unsigned int srcLane, int width = 32) {
	int2 a = *reinterpret_cast<int2*>(&var);
	a.x = __shfl_down_sync(0xffffffff, a.x, srcLane, width);
	a.y = __shfl_down_sync(0xffffffff, a.y, srcLane, width);
	return *reinterpret_cast<double*>(&a);
}

inline __device__ double __shfl_xor_double(double var, unsigned int srcLane, int width = 32) {
	int2 a = *reinterpret_cast<int2*>(&var);
	a.x = __shfl_xor_sync(0xffffffff, a.x, srcLane, width);
	a.y = __shfl_xor_sync(0xffffffff, a.y, srcLane, width);
	return *reinterpret_cast<double*>(&a);
}

inline __device__ double warpReduceSum_double(double val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
	return val;
}

inline __device__ double warpAllReduceSum_double(double val){
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__global__ void state_norm_gpu(double* ret, GTYPE *state, ITYPE dim){
	double sum = double(0.0);
	double real, imag;
    for (ITYPE index = blockIdx.x * blockDim.x + threadIdx.x; index < dim; index += blockDim.x * gridDim.x) {
		real = cuCreal(state[index]);
        imag = cuCimag(state[index]);
        sum += real*real+imag*imag;
	}
	sum = warpReduceSum_double(sum);
	
	if ((threadIdx.x & (warpSize - 1)) == 0){
        atomicAdd_double(ret, sum);
    }
}

__host__ double state_norm_cublas_host(void *state, ITYPE dim) {
    cublasStatus_t status;
    cublasHandle_t handle;
    double norm;
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);

    /* Initialize CUBLAS */
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    status = cublasDznrm2(handle, dim, state_gpu, 1, &norm);
	if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! cublasDznrm2 execution error.\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error\n");
        return EXIT_FAILURE;
    }
 
	state = reinterpret_cast<void*>(state_gpu);
    return norm;
}

__host__ double state_norm_host(void *state, ITYPE dim) {
    if(dim<=INT_MAX){
        return state_norm_cublas_host(state, dim);
    }

    cudaError_t cudaStatus;
    double norm;
    norm = state_norm_cublas_host(state, dim);
    double* norm_gpu;
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);

	checkCudaErrors(cudaMalloc((void**)&norm_gpu, sizeof(double)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemsetAsync(norm_gpu, 0, sizeof(double)), __FILE__, __LINE__);

	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
    
    state_norm_gpu <<< grid, block >>>(norm_gpu, state_gpu, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(&norm, norm_gpu, sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	checkCudaErrors(cudaFree(norm_gpu), __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
    return sqrt(norm);
}

__global__ void measurement_distribution_entropy_gpu(double* ret, const GTYPE *state, ITYPE dim){
	double sum = 0;
    const double eps = 1e-15;
	
    double prob;
    for (ITYPE index = blockIdx.x * blockDim.x + threadIdx.x; index < dim; index += blockDim.x * gridDim.x) {
		prob = cuCabs(state[index]);
        prob = prob * prob;
        if(prob > eps){
            sum += -1.0*prob*log(prob);
        } 
	}
	sum = warpReduceSum_double(sum);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0]), sum);
	}
}

__host__ double measurement_distribution_entropy_host(void* state, ITYPE dim){
	cudaError_t cudaStatus;
    double ent;
    double* ent_gpu;
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);

	checkCudaErrors(cudaMalloc((void**)&ent_gpu, sizeof(double)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemset(ent_gpu, 0, sizeof(double)), __FILE__, __LINE__);

	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
    
    measurement_distribution_entropy_gpu << <grid, block >> >(ent_gpu, state_gpu, dim);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(&ent, ent_gpu, sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	checkCudaErrors(cudaFree(ent_gpu), __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
    
    return ent;
}

__global__ void inner_product_gpu(GTYPE *ret, const GTYPE *psi, const GTYPE *phi, ITYPE dim){
	GTYPE sum = make_cuDoubleComplex(0.0, 0.0);
	for (ITYPE i = blockIdx.x * blockDim.x + threadIdx.x; i < dim; i += blockDim.x * gridDim.x) {
		sum = cuCadd(sum, cuCmul(cuConj(psi[i]), phi[i]));
	}
	sum.x = warpReduceSum_double(sum.x);
	sum.y = warpReduceSum_double(sum.y);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0].x), sum.x);
		atomicAdd_double(&(ret[0].y), sum.y);
	}
}

__host__ CPPCTYPE inner_product_cublas_host(const void *bra_state, const void *ket_state, ITYPE dim) {
	const GTYPE* bra_state_gpu = reinterpret_cast<const GTYPE*>(bra_state);
	const GTYPE* ket_state_gpu = reinterpret_cast<const GTYPE*>(ket_state);
    cublasStatus_t status;
    cublasHandle_t handle;
	GTYPE ret_g;
    CPPCTYPE ret;

    /* Initialize CUBLAS */
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

	status = cublasZdotc(handle, dim, bra_state_gpu, 1, ket_state_gpu, 1, &ret_g);
	if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! cublasZDotc execution error.\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error\n");
        return EXIT_FAILURE;
    }

	bra_state = reinterpret_cast<const void*>(bra_state_gpu);
	ket_state = reinterpret_cast<const void*>(ket_state_gpu);
    ret = CPPCTYPE(cuCreal(ret_g), cuCimag(ret_g));
	return ret;
}

__host__ CPPCTYPE inner_product_host(const void *bra_state, const void *ket_state, ITYPE dim) {
    if(dim<=INT_MAX){
        return inner_product_cublas_host(bra_state, ket_state, dim);
    }
    const GTYPE* bra_state_gpu = reinterpret_cast<const GTYPE*>(bra_state);
	const GTYPE* ket_state_gpu = reinterpret_cast<const GTYPE*>(ket_state);
	cudaError_t cudaStatus;
	CPPCTYPE ret=CPPCTYPE(0.0,0.0);
	GTYPE *ret_gpu;

	checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(GTYPE)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(ret_gpu, &ret, sizeof(GTYPE), cudaMemcpyHostToDevice), __FILE__, __LINE__);

	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	inner_product_gpu << <grid, block >> >(ret_gpu, bra_state_gpu, ket_state_gpu, dim);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(&ret, ret_gpu, sizeof(GTYPE), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	checkCudaErrors(cudaFree(ret_gpu), __FILE__, __LINE__);
	bra_state = reinterpret_cast<const void*>(bra_state_gpu);
	ket_state = reinterpret_cast<const void*>(ket_state_gpu);
	return ret;
}

__global__ void expectation_value_PauliX_gpu(double *ret, GTYPE *state, unsigned int target_qubit_index, ITYPE dim){
    double sum = 0.0;
    ITYPE basis0, basis1;
    ITYPE half_dim = dim>>1;
	for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x; state_index < half_dim; state_index += blockDim.x * gridDim.x) {
        basis0 = (state_index >> target_qubit_index);
        basis0 = basis0 << (target_qubit_index + 1);
        basis0 += state_index & ((1ULL << target_qubit_index) - 1);
        basis1 = basis0 ^ (1ULL << target_qubit_index);

        sum += cuCreal( cuCmul(cuConj(state[basis0]), state[basis1]) ) * 2;
	}
	sum = warpReduceSum_double(sum);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0]), sum);
    }
}

__global__ void expectation_value_PauliY_gpu(double *ret, GTYPE *state, unsigned int target_qubit_index, ITYPE dim){
	double sum = 0.0;
    ITYPE basis0, basis1;
    ITYPE half_dim = dim>>1;
	for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x; state_index < half_dim; state_index += blockDim.x * gridDim.x) {
        basis0 = (state_index >> target_qubit_index);
        basis0 = basis0 << (target_qubit_index + 1);
        basis0 += state_index & ((1ULL << target_qubit_index) - 1);
        basis1 = basis0 ^ (1ULL << target_qubit_index);
        sum += cuCimag( cuCmul(cuConj(state[basis0]), state[basis1]) ) * 2;
	}
	sum = warpReduceSum_double(sum);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0]), sum);
    }
}

__global__ void expectation_value_PauliZ_gpu(double *ret, GTYPE *state, unsigned int target_qubit_index, ITYPE dim){
    double sum=0.0;
    ITYPE loop_dim = dim;
	for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x; state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        int sign = 1 - 2 * ((state_index >> target_qubit_index) & 1);
        sum += cuCreal( cuCmul( cuConj(state[state_index]), state[state_index]) ) * sign;
	}
	sum = warpReduceSum_double(sum);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0]), sum);
	}
}

__global__ void expectation_value_single_qubit_Pauli_operator_gpu(
	GTYPE *ret, GTYPE *state, unsigned int target_qubit_index, ITYPE DIM
	){
	GTYPE sum = make_cuDoubleComplex(0.0, 0.0);
	GTYPE tmp;
	unsigned int j=0;
	for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x; state_index < DIM; state_index += blockDim.x * gridDim.x) {
		tmp = state[state_index];
		j = (state_index >> target_qubit_index) & 1;
		if (j){
			tmp = cuCadd(cuCmul(matrix_const_gpu[2], state[state_index^(1<<target_qubit_index)]), cuCmul(matrix_const_gpu[3], state[state_index]));
		}
		else{
			tmp = cuCadd(cuCmul(matrix_const_gpu[0], state[state_index]), cuCmul(matrix_const_gpu[1], state[state_index^(1<<target_qubit_index)]));
		}
		sum = cuCadd(sum, cuCmul(cuConj(state[state_index]), tmp));
	}
	sum.x = warpReduceSum_double(sum.x);
	sum.y = warpReduceSum_double(sum.y);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0].x), sum.x);
		atomicAdd_double(&(ret[0].y), sum.y);
	}
}

__host__ double expectation_value_single_qubit_Pauli_operator_host(unsigned int operator_index, unsigned int targetQubitIndex, GTYPE *psi_gpu, ITYPE dim) {
    double h_ret;
    double* d_ret;
	cuDoubleComplex PAULI_MATRIX[4][4] = {
		{ make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0) },
		{ make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0) },
		{ make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, -1), make_cuDoubleComplex(0, 1), make_cuDoubleComplex(0, 0) },
		{ make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(-1, 0) }
	};
	

    ITYPE half_dim = dim>>1;
	
    unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;

	checkCudaErrors(cudaMalloc((void**)&d_ret, sizeof(double)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(d_ret, &h_ret, sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    
    if(operator_index==1){
	    block = half_dim <= 1024 ? half_dim : 1024;
	    grid = half_dim / block;
        expectation_value_PauliX_gpu<< <grid, block>> >(d_ret, psi_gpu, targetQubitIndex, dim);
    }else if(operator_index==2){
	    block = half_dim <= 1024 ? half_dim : 1024;
	    grid = half_dim / block;
        expectation_value_PauliY_gpu<< <grid, block>> >(d_ret, psi_gpu, targetQubitIndex, dim);
    }else if(operator_index==3){
	    block = dim <= 1024 ? dim : 1024;
	    grid = dim / block;
        expectation_value_PauliZ_gpu<< <grid, block>> >(d_ret, psi_gpu, targetQubitIndex, dim);
    }else{
        // operator index=0
	    CPPCTYPE ret = CPPCTYPE(0.0, 0.0);
	    GTYPE *ret_gpu;
	    block = dim <= 1024 ? dim : 1024;
	    grid = dim / block;
	    checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(CPPCTYPE)), __FILE__, __LINE__);
	    checkCudaErrors(cudaMemcpy(ret_gpu, &ret, sizeof(CPPCTYPE), cudaMemcpyHostToDevice), __FILE__, __LINE__);
        checkCudaErrors(cudaMemcpyToSymbol(matrix_const_gpu, PAULI_MATRIX[operator_index], sizeof(GTYPE)*4), __FILE__, __LINE__);
	    expectation_value_single_qubit_Pauli_operator_gpu << <grid, block >> >(ret_gpu, psi_gpu, targetQubitIndex, dim);
	    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	    checkCudaErrors(cudaMemcpy(&ret, ret_gpu, sizeof(CPPCTYPE), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        checkCudaErrors(cudaFree(ret_gpu), __FILE__, __LINE__);
        checkCudaErrors(cudaFree(d_ret), __FILE__, __LINE__);
	    return ret.real();
    }
	
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(&h_ret, d_ret, sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_ret), __FILE__, __LINE__);
    return h_ret;
}

__device__ void multi_Z_gate_device(ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu)
{
	ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int minus_cnt = 0;
	if (idx < DIM){
		minus_cnt = __popcll(idx&bit_mask);
		if (minus_cnt & 1) psi_gpu[idx] = make_cuDoubleComplex(-psi_gpu[idx].x, -psi_gpu[idx].y);
	}
}

__global__ void multi_Z_gate_gpu(ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu)
{
	multi_Z_gate_device(bit_mask, DIM, psi_gpu);
}

__host__ void multi_Z_gate_host(int* gates, GTYPE *psi_gpu, ITYPE dim, int n_qubits){
	ITYPE bit_mask=0;
	for (int i = 0; i < n_qubits; ++i){
		if (gates[i]==3) bit_mask ^= (1 << i);
	}
	cudaError_t cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	multi_Z_gate_gpu << <grid, block >> >(bit_mask, dim, psi_gpu);
	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
}

__device__ GTYPE multi_Z_get_expectation_value_device(ITYPE idx, ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu)
{
	GTYPE ret=make_cuDoubleComplex(0.0,0.0);
	// ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int minus_cnt = 0;
	if (idx < DIM){
		GTYPE tmp_psi_gpu = psi_gpu[idx];
		minus_cnt = __popcll(idx&bit_mask);
		if (minus_cnt & 1) tmp_psi_gpu = make_cuDoubleComplex(-tmp_psi_gpu.x, -tmp_psi_gpu.y);
		ret = cuCmul(cuConj(psi_gpu[idx]), tmp_psi_gpu);
	}
	return ret;
}

__global__ void multi_Z_get_expectation_value_gpu(GTYPE *ret, ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu)
{
	GTYPE sum = make_cuDoubleComplex(0.0, 0.0);
	GTYPE tmp;
	for (ITYPE state = blockIdx.x * blockDim.x + threadIdx.x; state < DIM; state += blockDim.x * gridDim.x) {
		tmp = multi_Z_get_expectation_value_device(state, bit_mask, DIM, psi_gpu);
		sum = cuCadd(sum, tmp);
	}
	sum.x = warpReduceSum_double(sum.x);
	sum.y = warpReduceSum_double(sum.y);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0].x), sum.x);
		atomicAdd_double(&(ret[0].y), sum.y);
	}
}

__device__ GTYPE multipauli_get_expectation_value_device(ITYPE idx, ITYPE DIM, GTYPE *psi_gpu, int n_qubits){
	GTYPE ret;
	GTYPE tmp_psi, tmp_prev_state_psi, tmp_state_psi;
	ITYPE prev_state, state;
	int num_y1 = 0;
	int num_z1 = 0;
	int i_cnt = 0;
	int minus_cnt = 0;
	if (idx < DIM){
		i_cnt = num_pauli_op_gpu[2];
		num_y1 = __popcll(idx&bit_mask_gpu[2]);
		num_z1 = __popcll(idx&bit_mask_gpu[3]);
		minus_cnt = num_y1 + num_z1;
		prev_state = idx;
		state = idx^(bit_mask_gpu[1]+bit_mask_gpu[2]);
		tmp_prev_state_psi = psi_gpu[prev_state];
		tmp_state_psi = psi_gpu[state];
		// swap
		tmp_psi = tmp_state_psi;
		tmp_state_psi = tmp_prev_state_psi;
		tmp_prev_state_psi = tmp_psi;
		if (minus_cnt & 1) tmp_state_psi = make_cuDoubleComplex(-tmp_state_psi.x, -tmp_state_psi.y);
		if (i_cnt & 1) tmp_state_psi = make_cuDoubleComplex(tmp_state_psi.y, tmp_state_psi.x);
		if ((i_cnt >> 1) & 1) tmp_state_psi = make_cuDoubleComplex(-tmp_state_psi.x, -tmp_state_psi.y);
		// tmp_state      -> state      : state*conj(tmp_state)
		// tmp_prev_state -> prev_state : prev_state*conj(tmp_prev_state)
		ret = cuCmul(tmp_state_psi, cuConj(psi_gpu[state]));
	}
	return ret;
}

__global__ void multipauli_get_expectation_value_gpu(GTYPE* ret, ITYPE DIM, GTYPE *psi_gpu, int n_qubits){
	GTYPE sum = make_cuDoubleComplex(0.0, 0.0);
	GTYPE tmp;
	for (ITYPE state = blockIdx.x * blockDim.x + threadIdx.x; state < DIM; state += blockDim.x * gridDim.x) {
		tmp = multipauli_get_expectation_value_device(state, DIM, psi_gpu, n_qubits);
		sum = cuCadd(sum, tmp);
	}
	sum.x = warpReduceSum_double(sum.x);
	sum.y = warpReduceSum_double(sum.y);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0].x), sum.x);
		atomicAdd_double(&(ret[0].y), sum.y);
	}
}

__host__ double multipauli_get_expectation_value_host(unsigned int* gates, GTYPE *psi_gpu, ITYPE DIM, int n_qubits){
	CPPCTYPE ret[1];
	ret[0]=CPPCTYPE(0,0);
	GTYPE *ret_gpu;

	checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(GTYPE)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(ret_gpu, ret, sizeof(GTYPE), cudaMemcpyHostToDevice), __FILE__, __LINE__);

	unsigned int num_pauli_op[4] = { 0, 0, 0, 0 };
	for (int i = 0; i < n_qubits; ++i) ++num_pauli_op[gates[i]];
	ITYPE bit_mask[4] = { 0, 0, 0, 0 };
	for (int i = 0; i < n_qubits; ++i){
		bit_mask[gates[i]] ^= (1 << i);
	}
	if (num_pauli_op[1] == 0 && num_pauli_op[2] == 0){
		unsigned int block = DIM <= 1024 ? DIM : 1024;
		unsigned int grid = DIM / block;
		multi_Z_get_expectation_value_gpu << <grid, block >> >(ret_gpu, bit_mask[3], DIM, psi_gpu);
		checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
		checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
		checkCudaErrors(cudaMemcpy(ret, ret_gpu, sizeof(CPPCTYPE), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
		checkCudaErrors(cudaFree(ret_gpu), __FILE__, __LINE__);
		return ret[0].real();
	}
	
        checkCudaErrors(cudaMemcpyToSymbol(num_pauli_op_gpu, num_pauli_op, sizeof(unsigned int)*4), __FILE__, __LINE__);
        checkCudaErrors(cudaMemcpyToSymbol(bit_mask_gpu, bit_mask, sizeof(ITYPE)*4), __FILE__, __LINE__);

	
	unsigned int block = DIM <= 1024 ? DIM : 1024;
	unsigned int grid = DIM / block;
	multipauli_get_expectation_value_gpu << <grid, block >> >(ret_gpu, DIM, psi_gpu, n_qubits);
	
	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(ret, ret_gpu, sizeof(CPPCTYPE), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(ret_gpu), __FILE__, __LINE__);
	return ret[0].real();
}

// calculate probability with which we obtain 0 at target qubit
__global__ void M0_prob_gpu(double* ret, UINT target_qubit_index, const GTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim>>1;
    double sum =0.;
    double tmp;
	for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x; state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        ITYPE basis_0 = insert_zero_to_basis_index_device(state_index, target_qubit_index);
        tmp = cuCabs(state[basis_0]);
        sum += tmp*tmp;
	}
	sum = warpReduceSum_double(sum);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0]), sum);
	}
}

// calculate probability with which we obtain 0 at target qubit
__host__ double M0_prob_host(UINT target_qubit_index, void* state, ITYPE dim){
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    double ret[1]={0.0};
	double *ret_gpu;

	checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(double)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(ret_gpu, ret, sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);

	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	
    M0_prob_gpu << <grid, block >> >(ret_gpu, target_qubit_index, state_gpu, dim);
	
	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(ret, ret_gpu, sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(ret_gpu), __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
	return ret[0];
}

// calculate probability with which we obtain 1 at target qubit
__global__ void M1_prob_gpu(double* ret, UINT target_qubit_index, const GTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim>>1;
    const ITYPE mask = 1ULL << target_qubit_index;
    double sum =0.;
    double tmp;
	for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x; state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        ITYPE basis_1 = insert_zero_to_basis_index_device(state_index, target_qubit_index) ^ mask;
        tmp = cuCabs(state[basis_1]);
        sum += tmp*tmp;
	}
	sum = warpReduceSum_double(sum);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0]), sum);
	}
}

__host__ double M1_prob_host(UINT target_qubit_index, void* state, ITYPE dim){
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    double ret[1]={0.0};
	double *ret_gpu;

	checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(double)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(ret_gpu, ret, sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);

	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	
    M1_prob_gpu << <grid, block >> >(ret_gpu, target_qubit_index, state_gpu, dim);
	
	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(ret, ret_gpu, sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(ret_gpu), __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
	return ret[0];
}


// calculate merginal probability with which we obtain the set of values measured_value_list at sorted_target_qubit_index_list
// warning: sorted_target_qubit_index_list must be sorted.
__global__ void marginal_prob_gpu(double* ret_gpu, const UINT* sorted_target_qubit_index_list, const UINT* measured_value_list, UINT target_qubit_index_count, const GTYPE* state, ITYPE dim){
    ITYPE loop_dim = dim >> target_qubit_index_count;
    double sum =0.;
    double tmp;
	for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x; state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        ITYPE basis = state_index;
        for(UINT cursor=0; cursor < target_qubit_index_count ; cursor++){
            UINT insert_index = sorted_target_qubit_index_list[cursor];
            ITYPE mask = 1ULL << insert_index;
            basis = insert_zero_to_basis_index_device(basis, insert_index );
            basis ^= mask * measured_value_list[cursor];
        }
        tmp = cuCabs(state[basis]);
        sum += tmp*tmp;
	}
	sum = warpReduceSum_double(sum);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret_gpu[0]), sum);
	}
}

__host__ double marginal_prob_host(UINT* sorted_target_qubit_index_list, UINT* measured_value_list, UINT target_qubit_index_count, void* state, ITYPE dim){
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    double ret[1]={0.0};
	double *ret_gpu;
    UINT* sorted_target_qubit_index_list_gpu;
    UINT* measured_value_list_gpu; 
	
    checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(double)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(ret_gpu, ret, sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&sorted_target_qubit_index_list_gpu, sizeof(UINT)*target_qubit_index_count), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(sorted_target_qubit_index_list_gpu, sorted_target_qubit_index_list, sizeof(UINT)*target_qubit_index_count, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    checkCudaErrors(cudaMalloc((void**)&measured_value_list_gpu, sizeof(UINT)*target_qubit_index_count), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(measured_value_list_gpu, measured_value_list, sizeof(UINT)*target_qubit_index_count, cudaMemcpyHostToDevice), __FILE__, __LINE__);

	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	
    marginal_prob_gpu << <grid, block >> >(ret_gpu, sorted_target_qubit_index_list_gpu, measured_value_list_gpu, target_qubit_index_count, state_gpu, dim);
	
	checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(ret, ret_gpu, sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(ret_gpu), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(sorted_target_qubit_index_list_gpu), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(measured_value_list_gpu), __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
	return ret[0];
}

__global__ void expectation_value_multi_qubit_Pauli_operator_XZ_mask_gpu(double* ret_gpu, ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,UINT pivot_qubit_index, GTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim>>1;
    double sum = 0.;
    double tmp;
	for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x; state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        ITYPE basis_0 = insert_zero_to_basis_index_device(state_index, pivot_qubit_index);
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;
        UINT sign_0 = __popcll(basis_0 & phase_flip_mask)&1;
        
        tmp = cuCreal(cuCmul( cuCmul(state[basis_0], cuConj(state[basis_1])), PHASE_90ROT_gpu[ (global_phase_90rot_count + sign_0*2)&3 ]))*2.0;
        sum += tmp;
    }
	sum = warpReduceSum_double(sum);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret_gpu[0]), sum);
	}
}

__host__ double expectation_value_multi_qubit_Pauli_operator_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,UINT pivot_qubit_index, void* state, ITYPE dim){
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError_t cudaStatus;
    double ret;
    double* ret_gpu;
    CPPCTYPE PHASE_90ROT[4] = {
        CPPCTYPE(1.0, 0.0), 
        CPPCTYPE(0.0, 1.0), 
        CPPCTYPE(-1.0,0.0), 
        CPPCTYPE(0.0, -1.0)};

    checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(double)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemset(ret_gpu, 0, sizeof(double)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpyToSymbol(PHASE_90ROT_gpu, PHASE_90ROT, sizeof(GTYPE)*4), __FILE__, __LINE__);

    const ITYPE loop_dim = dim>>1;
    
    unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;
	expectation_value_multi_qubit_Pauli_operator_XZ_mask_gpu<< <grid, block >> >(ret_gpu, bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_gpu, dim);
	
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(&ret, ret_gpu, sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(ret_gpu), __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
    
    return ret;
}

__global__ void expectation_value_multi_qubit_Pauli_operator_Z_mask_gpu(double* ret_gpu, ITYPE phase_flip_mask, const GTYPE* state, ITYPE dim){
    const ITYPE loop_dim = dim;
    double sum = 0.;
    double tmp;
	for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x; state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        UINT bit_parity = __popcll(state_index & phase_flip_mask)&1;
        int sign = 1 - 2*bit_parity;
        tmp = cuCabs(state[state_index]);
        sum += tmp * tmp * sign;
    }
	sum = warpReduceSum_double(sum);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret_gpu[0]), sum);
	}
}

__host__ double expectation_value_multi_qubit_Pauli_operator_Z_mask_host(ITYPE phase_flip_mask, void* state, ITYPE dim){
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError_t cudaStatus;
    double ret;
    double* ret_gpu;

    checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(double)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemset(ret_gpu, 0, sizeof(double)), __FILE__, __LINE__);

    const ITYPE loop_dim = dim>>1;
    
    unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;
	expectation_value_multi_qubit_Pauli_operator_Z_mask_gpu<< <grid, block >> >(ret_gpu, phase_flip_mask, state_gpu, dim);
	
    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(&ret, ret_gpu, sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(ret_gpu), __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
    
    return ret;
}

__host__ double expectation_value_multi_qubit_Pauli_operator_partial_list_host(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, void* state, ITYPE dim){
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list_gsim(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    double result;
    if(bit_flip_mask == 0){
        result = expectation_value_multi_qubit_Pauli_operator_Z_mask_host(phase_flip_mask, state,dim);
    }else{
        result = expectation_value_multi_qubit_Pauli_operator_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim);
    }
    return result;
}

__host__ double expectation_value_multi_qubit_Pauli_operator_whole_list_host(const UINT* Pauli_operator_type_list, UINT qubit_count, void* state, ITYPE dim){
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list_gsim(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    double result;
    if(bit_flip_mask == 0){
        result = expectation_value_multi_qubit_Pauli_operator_Z_mask_host(phase_flip_mask, state, dim);
    }else{
        result = expectation_value_multi_qubit_Pauli_operator_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state, dim);
    }
    return result;
}

__global__ void transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_gpu(GTYPE* ret_gpu, ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, const GTYPE* state_bra, const GTYPE* state_ket, ITYPE dim) {
	const ITYPE loop_dim = dim >> 1;

	GTYPE sum = make_cuDoubleComplex(0.0, 0.0);
    GTYPE tmp;
	for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x; state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
		ITYPE basis_0 = insert_zero_to_basis_index_device(state_index, pivot_qubit_index);
		ITYPE basis_1 = basis_0 ^ bit_flip_mask;
		
		UINT sign_0 = __popcll(basis_0 & phase_flip_mask) & 1;
	    tmp = cuCmul( cuCmul(state_ket[basis_0], cuConj(state_bra[basis_1])), PHASE_90ROT_gpu[(global_phase_90rot_count + sign_0 * 2) & 3 ]);
        sum = cuCadd(sum, tmp);

		UINT sign_1 = __popcll(basis_1 & phase_flip_mask) & 1;
		tmp = cuCmul( cuCmul(state_ket[basis_1], cuConj(state_bra[basis_0])), PHASE_90ROT_gpu[(global_phase_90rot_count + sign_1 * 2) & 3]);
        sum = cuCadd(sum, tmp);
	}
	sum.x = warpReduceSum_double(sum.x);
	sum.y = warpReduceSum_double(sum.y);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret_gpu[0].x), sum.x);
		atomicAdd_double(&(ret_gpu[0].y), sum.y);
	}
}

__host__ CPPCTYPE transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, void* state_bra, void* state_ket, ITYPE dim) {
	cudaError_t cudaStatus;
    GTYPE* state_bra_gpu = reinterpret_cast<GTYPE*>(state_bra);
    GTYPE* state_ket_gpu = reinterpret_cast<GTYPE*>(state_ket);
    CPPCTYPE ret;
    GTYPE* ret_gpu;

    checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(GTYPE)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemset(ret_gpu, 0, sizeof(GTYPE)), __FILE__, __LINE__);

    const ITYPE loop_dim = dim>>1;
    
    unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;
    transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_gpu<< <grid, block >> >(ret_gpu, bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_bra_gpu, state_ket_gpu, dim);

    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(&ret, ret_gpu, sizeof(GTYPE), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(ret_gpu), __FILE__, __LINE__);
	state_bra = reinterpret_cast<void*>(state_bra_gpu);
	state_ket = reinterpret_cast<void*>(state_ket_gpu);
    
    return ret;
}


__global__ void transition_amplitude_multi_qubit_Pauli_operator_Z_mask_gpu(GTYPE* ret, ITYPE phase_flip_mask, GTYPE* state_bra, GTYPE* state_ket, ITYPE dim) {
	const ITYPE loop_dim = dim;
	GTYPE sum = make_cuDoubleComplex(0.0, 0.0);
    GTYPE tmp;
	for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x; state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
		UINT bit_parity = __popcll(state_index & phase_flip_mask) & 1;
		double sign = 1 - 2 * bit_parity;
		tmp = cuCmul( make_cuDoubleComplex(sign,0.0), cuCmul( state_ket[state_index], cuConj(state_bra[state_index])));
        sum = cuCadd(sum, tmp);
    }
	sum.x = warpReduceSum_double(sum.x);
	sum.y = warpReduceSum_double(sum.y);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0].x), sum.x);
		atomicAdd_double(&(ret[0].y), sum.y);
	}
}

__host__ CPPCTYPE transition_amplitude_multi_qubit_Pauli_operator_Z_mask_host(ITYPE phase_flip_mask, void* state_bra, void* state_ket, ITYPE dim) {
	cudaError_t cudaStatus;
    GTYPE* state_bra_gpu = reinterpret_cast<GTYPE*>(state_bra);
    GTYPE* state_ket_gpu = reinterpret_cast<GTYPE*>(state_ket);
    CPPCTYPE ret;
    GTYPE* ret_gpu;

    checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(GTYPE)), __FILE__, __LINE__);
	checkCudaErrors(cudaMemset(ret_gpu, 0, sizeof(GTYPE)), __FILE__, __LINE__);

    const ITYPE loop_dim = dim;
    
    unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;
    transition_amplitude_multi_qubit_Pauli_operator_Z_mask_gpu<< <grid, block >> >(ret_gpu, phase_flip_mask, state_bra_gpu, state_ket_gpu, dim);

    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	checkCudaErrors(cudaMemcpy(&ret, ret_gpu, sizeof(GTYPE), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	checkCudaErrors(cudaFree(ret_gpu), __FILE__, __LINE__);
	state_bra = reinterpret_cast<void*>(state_bra_gpu);
	state_ket = reinterpret_cast<void*>(state_ket_gpu);
 
	return ret;
}

__host__ CPPCTYPE transition_amplitude_multi_qubit_Pauli_operator_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, void* state_bra, void* state_ket, ITYPE dim) {
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_partial_list_gsim(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	CPPCTYPE result;
	if (bit_flip_mask == 0) {
		result = transition_amplitude_multi_qubit_Pauli_operator_Z_mask_host(phase_flip_mask, state_bra, state_ket, dim);
	}
	else {
		result = transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_bra, state_ket, dim);
	}
	return result;
}

__host__ CPPCTYPE transition_amplitude_multi_qubit_Pauli_operator_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, void* state_bra, void* state_ket, ITYPE dim) {
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_whole_list_gsim(Pauli_operator_type_list, qubit_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	CPPCTYPE result;
	if (bit_flip_mask == 0) {
		result = transition_amplitude_multi_qubit_Pauli_operator_Z_mask_host(phase_flip_mask, state_bra, state_ket, dim);
	}
	else {
		result = transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_bra, state_ket, dim);
	}
	return result;
}

