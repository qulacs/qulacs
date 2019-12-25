#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "cuda.h"
// for using cublas 
#include <cublas_v2.h>

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <assert.h>
#include <algorithm>
#include <cuComplex.h>
#include "util_type.h"
#include "util.cuh"
#include "util_func.h"
#include "memory_ops.h"

int get_num_device(){
    int n_gpu;
    cudaGetDeviceCount(&n_gpu);
    return n_gpu;
}

void set_device(unsigned int device_num){
    cudaSetDevice(device_num);
}

int get_current_device() {
	int curr_dev_num;
	cudaGetDevice(&curr_dev_num);
	return curr_dev_num;
}
inline __device__ double __shfl_down_double(double var, unsigned int srcLane, int width = 32) {
	int2 a = *reinterpret_cast<int2*>(&var);
	a.x = __shfl_down_sync(a.x, srcLane, width);
	a.y = __shfl_down_sync(a.y, srcLane, width);
	return *reinterpret_cast<double*>(&a);
}

inline __device__ int warpReduceSum(int val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down_sync(0xffffffff, val, offset);
		// val += __shfl_down(val, offset);
	return val;
}

// __device__ int __popcll ( unsigned long long int x )
inline __device__ int popcount64(ITYPE b) {
    return __popcll(b);
}

//__device__ int __popc ( unsigned int  x )
inline __device__ int popcount32(unsigned int b) {
	return __popc(b);
}

__global__ void deviceReduceWarpAtomicKernel(int *in, int* out, ITYPE N) {
	int sum = int(0);
	for (ITYPE i = blockIdx.x * blockDim.x + threadIdx.x;
		i < N;
		i += blockDim.x * gridDim.x) {
		sum += in[i];
	}
	sum = warpReduceSum(sum);
	if ((threadIdx.x & (warpSize - 1)) == 0)
		atomicAdd(out, sum);
}

__global__ void set_computational_basis_gpu(ITYPE comp_basis, GTYPE* state, ITYPE dim){
 	ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < dim) {
		state[idx] = make_cuDoubleComplex(0.0, 0.0);
	}
    if(idx==comp_basis) state[comp_basis] = make_cuDoubleComplex(1.0, 0.0);
}

__host__ void set_computational_basis_host(ITYPE comp_basis, void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);

	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;

	set_computational_basis_gpu << <grid, block,0 , *cuda_stream >> > (comp_basis, state_gpu, dim);

	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);

	state = reinterpret_cast<void*>(state_gpu);
}

// copy state_gpu to state_gpu_copy
void copy_quantum_state_from_device_to_device(void* state_gpu_copy, const void* state_gpu, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	const GTYPE* psi_gpu = reinterpret_cast<const GTYPE*>(state_gpu);
	GTYPE* psi_gpu_copy = reinterpret_cast<GTYPE*>(state_gpu_copy);
	checkCudaErrors(cudaMemcpyAsync(psi_gpu_copy, psi_gpu, dim * sizeof(GTYPE), cudaMemcpyDeviceToDevice, *cuda_stream), __FILE__, __LINE__);
	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	state_gpu = reinterpret_cast<const void*>(psi_gpu);
	state_gpu_copy = reinterpret_cast<void*>(psi_gpu_copy);
}

// copy cppstate to state_gpu_copy
void copy_quantum_state_from_host_to_device(void* state_gpu_copy, const void* state, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
    GTYPE* psi_gpu_copy = reinterpret_cast<GTYPE*>(state_gpu_copy);
	checkCudaErrors(cudaMemcpyAsync(psi_gpu_copy, state, dim * sizeof(GTYPE), cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	state_gpu_copy = reinterpret_cast<void*>(psi_gpu_copy);
}

// this function will be removed in the future version
void copy_quantum_state_from_cppstate_host(void* state_gpu_copy, const CPPCTYPE* cppstate, ITYPE dim, void* stream, UINT device_number) {
	copy_quantum_state_from_host_to_device(state_gpu_copy, cppstate, dim, stream, device_number);
}

void copy_quantum_state_from_device_to_host(void* state_cpu_copy, const void* state_gpu_original, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);

	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	const GTYPE* psi_gpu = reinterpret_cast<const GTYPE*>(state_gpu_original);
	checkCudaErrors(cudaMemcpyAsync(state_cpu_copy, psi_gpu, dim * sizeof(GTYPE), cudaMemcpyDeviceToHost, *cuda_stream), __FILE__, __LINE__);
	checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
	state_gpu_original = reinterpret_cast<const void*>(psi_gpu);
}

// copy state_gpu to psi_cpu_copy
// this function is same as copy_quantum_state_from_device_to_host
void get_quantum_state_host(void* state_gpu, void* psi_cpu_copy, ITYPE dim, void* stream, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);
	cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
	GTYPE* psi_gpu = reinterpret_cast<GTYPE*>(state_gpu);
	psi_cpu_copy = reinterpret_cast<CPPCTYPE*>(psi_cpu_copy);
	checkCudaErrors(cudaMemcpyAsync(psi_cpu_copy, psi_gpu, dim * sizeof(CPPCTYPE), cudaMemcpyDeviceToHost, *cuda_stream), __FILE__, __LINE__);
	state_gpu = reinterpret_cast<void*>(psi_gpu);
}

void print_quantum_state_host(void* state, ITYPE dim, unsigned int device_number) {
	int current_device = get_current_device();
	if (device_number != current_device) cudaSetDevice(device_number);
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	CPPCTYPE* state_cpu=(CPPCTYPE*)malloc(sizeof(CPPCTYPE)*dim);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, state_gpu, dim * sizeof(CPPCTYPE), cudaMemcpyDeviceToHost));
	for(int i=0;i<dim;++i){
		std::cout << i << " : " << state_cpu[i].real() << "+i" << state_cpu[i].imag() << '\n'; 
	}
	std::cout << '\n';
	free(state_cpu);
	state = reinterpret_cast<void*>(state);
}

ITYPE insert_zero_to_basis_index_gsim(ITYPE basis_index, unsigned int qubit_index){
    ITYPE temp_basis = (basis_index >> qubit_index) << (qubit_index+1);
    return temp_basis + (basis_index & ( (1ULL<<qubit_index) -1));
}

void get_Pauli_masks_partial_list_gsim(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count,
    ITYPE* bit_flip_mask, ITYPE* phase_flip_mask, UINT* global_phase_90rot_count, UINT* pivot_qubit_index){
    (*bit_flip_mask)=0;
    (*phase_flip_mask)=0;
    (*global_phase_90rot_count)=0;
    (*pivot_qubit_index)=0;
    for(UINT cursor=0;cursor < target_qubit_index_count; ++cursor){
        UINT target_qubit_index = target_qubit_index_list[cursor];
        switch(Pauli_operator_type_list[cursor]){
        case 0: // I
            break;
        case 1: // X
            (*bit_flip_mask) ^= 1ULL << target_qubit_index;
            (*pivot_qubit_index) = target_qubit_index;
            break;
        case 2: // Y
            (*bit_flip_mask) ^= 1ULL << target_qubit_index;
            (*phase_flip_mask) ^= 1ULL << target_qubit_index;
            (*global_phase_90rot_count) ++;
            (*pivot_qubit_index) = target_qubit_index;
            break;
        case 3: // Z
            (*phase_flip_mask) ^= 1ULL << target_qubit_index;
            break;
        default:
            fprintf(stderr,"Invalid Pauli operator ID called");
            assert(0);
        }
    }
}

void get_Pauli_masks_whole_list_gsim(const UINT* Pauli_operator_type_list, UINT target_qubit_index_count,
    ITYPE* bit_flip_mask, ITYPE* phase_flip_mask, UINT* global_phase_90rot_count, UINT* pivot_qubit_index){

    (*bit_flip_mask)=0;
    (*phase_flip_mask)=0;
    (*global_phase_90rot_count)=0;
    (*pivot_qubit_index)=0;
    for(UINT target_qubit_index=0; target_qubit_index < target_qubit_index_count; ++target_qubit_index){
        switch(Pauli_operator_type_list[target_qubit_index]){
        case 0: // I
            break;
        case 1: // X
            (*bit_flip_mask) ^= 1ULL << target_qubit_index;
            (*pivot_qubit_index) = target_qubit_index;
            break;
        case 2: // Y
            (*bit_flip_mask) ^= 1ULL << target_qubit_index;
            (*phase_flip_mask) ^= 1ULL << target_qubit_index;
            (*global_phase_90rot_count) ++;
            (*pivot_qubit_index) = target_qubit_index;
            break;
        case 3: // Z
            (*phase_flip_mask) ^= 1ULL << target_qubit_index;
            break;
        default:
            fprintf(stderr,"Invalid Pauli operator ID called");
            assert(0);
        }
    }
}

ITYPE* create_matrix_mask_list_gsim(const UINT* qubit_index_list, UINT qubit_index_count){
    const ITYPE matrix_dim = 1ULL << qubit_index_count;
    ITYPE* mask_list = (ITYPE*) calloc((size_t)matrix_dim, sizeof(ITYPE));
    ITYPE cursor = 0;

    for(cursor=0;cursor < matrix_dim; ++cursor){
        for(UINT bit_cursor = 0; bit_cursor < qubit_index_count;++bit_cursor){
			if ((cursor >> bit_cursor) & 1) {
				UINT bit_index = qubit_index_list[bit_cursor];
				mask_list[cursor] ^= (1ULL << bit_index);
			}
        }
    }
    return mask_list;
}

ITYPE create_control_mask_gsim(const UINT* qubit_index_list, const UINT* value_list, UINT size) {
	ITYPE mask = 0;
	for (UINT cursor = 0; cursor < size; ++cursor) {
		mask ^= (1ULL << qubit_index_list[cursor]) * value_list[cursor];
	}
	return mask;
}

UINT* create_sorted_ui_list_gsim(const UINT* array, size_t size){
    UINT* new_array = (UINT*)calloc(size,sizeof(UINT));
    memcpy(new_array, array, size*sizeof(UINT));
    std::sort(new_array, new_array+size);
    return new_array;
}

UINT* create_sorted_ui_list_value_gsim(const UINT* array, size_t size, UINT value){
    UINT* new_array = (UINT*)calloc(size+1, sizeof(UINT));
    memcpy(new_array,array,size*sizeof(UINT));
    new_array[size]=value;
    std::sort(new_array, new_array+size+1);
    return new_array;
}

UINT* create_sorted_ui_list_list_gsim(const UINT* array1, size_t size1, const UINT* array2, size_t size2){
    UINT* new_array = (UINT*)calloc(size1+size2, sizeof(UINT));
    memcpy(new_array,array1,size1*sizeof(UINT));
    memcpy(new_array+size1,array2,size2*sizeof(UINT));
    std::sort(new_array, new_array+size1+size2);
    return new_array;
}

// C=alpha*A*B+beta*C
// in this wrapper, we assume beta is always zero!
int cublas_zgemm_wrapper(ITYPE n, CPPCTYPE alpha, const CPPCTYPE *h_A, const CPPCTYPE *h_B, CPPCTYPE beta, CPPCTYPE *h_C){
    ITYPE n2 = n*n;
    cublasStatus_t status;
    cublasHandle_t handle;
    GTYPE *d_A;// = make_cuDoubleComplex(0.0,0.0);
    GTYPE *d_B;// = make_cuDoubleComplex(0,0);
    GTYPE *d_C;// = make_cuDoubleComplex(0,0);
    GTYPE d_alpha=make_cuDoubleComplex(alpha.real(), alpha.imag());
    GTYPE d_beta=make_cuDoubleComplex(beta.real(), beta.imag());
    // int dev = 0; //findCudaDevice(argc, (const char **)argv);
    
    /* Initialize CUBLAS */
    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    /* Allocate device memory for the matrices */
    if (cudaMalloc(reinterpret_cast<void **>(&d_A), n2 * sizeof(d_A[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc(reinterpret_cast<void **>(&d_B), n2 * sizeof(d_B[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc(reinterpret_cast<void **>(&d_C), n2 * sizeof(d_C[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    //status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
    status = cublasSetMatrix(n, n, sizeof(h_A[0]), h_A, n, d_A, n);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }

    //status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
    status = cublasSetMatrix(n, n, sizeof(h_B[0]), h_B, n, d_B, n);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }

    //status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);
    status = cublasSetMatrix(n, n, sizeof(h_C[0]), h_C, n, d_C, n);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }
    /* Performs operation using cublas */
    status = cublasZgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &d_alpha, d_A,
        n, d_B, n, &d_beta, d_C, n);

    //status=cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    /* Allocate host memory for reading back the result from device memory */
    CPPCTYPE* tmp_h_C = reinterpret_cast<CPPCTYPE *>(malloc(n2 * sizeof(h_C[0])));

    if (tmp_h_C == 0) {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Read the result back */
    status = cublasGetMatrix(n, n, sizeof(GTYPE), d_C, n, tmp_h_C, n);
    memcpy(h_C, tmp_h_C, sizeof(h_C[0])*n2);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }
    if (cudaFree(d_A) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_B) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_C) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }
    return 0;
}

// C=alpha*A*x+beta*y
// in this wrapper, we assume beta is always zero!
int cublas_zgemv_wrapper(ITYPE n, CPPCTYPE alpha, const CPPCTYPE *h_A, const CPPCTYPE *h_x, CPPCTYPE beta, CPPCTYPE *h_y){
    ITYPE n2 = n*n;
    cublasStatus_t status;
    cublasHandle_t handle;
    GTYPE *d_A;
    GTYPE *d_x;
    GTYPE *d_y;
    GTYPE d_alpha=make_cuDoubleComplex(alpha.real(), alpha.imag());
    GTYPE d_beta=make_cuDoubleComplex(beta.real(), beta.imag());
    // int dev = 0; //findCudaDevice(argc, (const char **)argv);
    
    /* Initialize CUBLAS */
    printf("simpleCUBLAS test running..\n");
    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    /* Allocate device memory for the matrices */
    if (cudaMalloc(reinterpret_cast<void **>(&d_A), n2 * sizeof(d_A[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc(reinterpret_cast<void **>(&d_x), n * sizeof(d_x[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate x)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc(reinterpret_cast<void **>(&d_y), n * sizeof(d_y[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate y)\n");
        return EXIT_FAILURE;
    }

    /* Initialize the device matrices with the host matrices */
    //status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
    status = cublasSetMatrix(n, n, sizeof(h_A[0]), h_A, n, d_A, n);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n, sizeof(h_x[0]), h_x, 1, d_x, 1);
    //status = cublasSetMatrix(n, n, sizeof(h_B[0]), h_B, n, d_B, n);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write x)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n, sizeof(h_y[0]), h_y, 1, d_y, 1);
    //status = cublasSetMatrix(n, n, sizeof(h_C[0]), h_C, n, d_C, n);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }
    /* Performs operation using cublas */
    status = cublasZgemv(handle, CUBLAS_OP_T, n, n, &d_alpha, d_A, n,
        d_x, 1, &d_beta, d_y, 1);
/*
cublasStatus_t cublasZgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *beta,
                           cuDoubleComplex *y, int incy)
*/
    //status=cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    /* Allocate host memory for reading back the result from device memory */
    CPPCTYPE* tmp_h_y = reinterpret_cast<CPPCTYPE *>(malloc(n * sizeof(h_y[0])));

    if (tmp_h_y == 0) {
        fprintf(stderr, "!!!! host memory allocation error (y)\n");
        return EXIT_FAILURE;
    }

    /* Read the result back */
    status = cublasGetVector(n, sizeof(GTYPE), d_y, 1, tmp_h_y, 1);
    /*
    cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy)
    */
    memcpy(h_y, tmp_h_y, sizeof(h_y[0])*n);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }
    if (cudaFree(d_A) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_x) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (x)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_y) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (y)\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }
    return 0;
}

// we assume state has already allocated at device
int cublas_zgemv_wrapper(ITYPE n, const CPPCTYPE *h_matrix, GTYPE *d_state){
    ITYPE n2 = n*n;
    cublasStatus_t status;
    cublasHandle_t handle;
    GTYPE *d_matrix;
    GTYPE *d_y; // this will include the answer of the state.
    GTYPE d_alpha = make_cuDoubleComplex(1.0, 0.0);
    GTYPE d_beta = make_cuDoubleComplex(0.0, 0.0);
    // int dev = 0;
    
    /* Initialize CUBLAS */
    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    /* Allocate device memory for the matrices */
    if (cudaMalloc(reinterpret_cast<void **>(&d_matrix), n2 * sizeof(d_matrix[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc(reinterpret_cast<void **>(&d_y), n * sizeof(d_y[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate y)\n");
        return EXIT_FAILURE;
    }
    // cudaMemset(&d_y, 0, sizeof(d_y[0])*n);
    /* Initialize the device matrices with the host matrices */
    status = cublasSetMatrix(n, n, sizeof(h_matrix[0]), h_matrix, n, d_matrix, n);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }

    /* Performs operation using cublas */
    status = cublasZgemv(handle, CUBLAS_OP_T, n, n, &d_alpha, d_matrix, n,
        d_state, 1, &d_beta, d_y, 1);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    cudaMemcpy(d_state, d_y, n * sizeof(GTYPE), cudaMemcpyDeviceToDevice);
   
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }
    if (cudaFree(d_matrix) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_y) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (y)\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }
    return 0;
}


// we assume state and matrix has already allocated at device
int cublas_zgemv_wrapper(ITYPE n, const GTYPE *d_matrix, GTYPE *d_state){
    // ITYPE n2 = n*n;
    cublasStatus_t status;
    cublasHandle_t handle;
    GTYPE *d_y; // this will include the answer of the state.
    GTYPE d_alpha = make_cuDoubleComplex(1.0, 0.0);
    GTYPE d_beta = make_cuDoubleComplex(0.0, 0.0);
    // int dev = 0;
    
    /* Initialize CUBLAS */
    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc(reinterpret_cast<void **>(&d_y), n * sizeof(d_y[0])) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate y)\n");
        return EXIT_FAILURE;
    }
    // cudaMemset(&d_y, 0, sizeof(d_y[0])*n);

    /* Performs operation using cublas */
    status = cublasZgemv(handle, CUBLAS_OP_T, n, n, &d_alpha, d_matrix, n,
        d_state, 1, &d_beta, d_y, 1);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    cudaMemcpy(d_state, d_y, n * sizeof(GTYPE), cudaMemcpyDeviceToDevice);
   
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_y) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (y)\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }
    return 0;
}
