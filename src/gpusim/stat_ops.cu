#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpu_wrapping.h"
// #include <sys/time.h>

#include <limits.h>

#include "stat_ops.h"
#include "stat_ops_device_functions.h"
#include "update_ops_cuda.h"
#include "util.cuh"
#include "util_func.h"
#include "util_type.h"

__constant__ GTYPE matrix_const_gpu[4];
__constant__ unsigned int num_pauli_op_gpu[4];
__constant__ ITYPE bit_mask_gpu[4];
__constant__ GTYPE PHASE_90ROT_gpu[4];

__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

inline __device__ double __shfl_down_double(
#ifdef __HIP_PLATFORM_AMD__
    double var, unsigned int srcLane, int width = 64) {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
#else
    double var, unsigned int srcLane, int width = 32) {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down_sync(0xffffffff, a.x, srcLane, width);
    a.y = __shfl_down_sync(0xffffffff, a.y, srcLane, width);
#endif
    return *reinterpret_cast<double*>(&a);
}

inline __device__ double __shfl_xor_double(
#ifdef __HIP_PLATFORM_AMD__
    double var, unsigned int srcLane, int width = 64) {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_xor(a.x, srcLane, width);
    a.y = __shfl_xor(a.y, srcLane, width);
#else
    double var, unsigned int srcLane, int width = 32) {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_xor_sync(0xffffffff, a.x, srcLane, width);
    a.y = __shfl_xor_sync(0xffffffff, a.y, srcLane, width);
#endif
    return *reinterpret_cast<double*>(&a);
}

inline __device__ double warpReduceSum_double(double val) {
#pragma unroll
    for (int offset = (warpSize >> 1); offset > 0; offset >>= 1)
#ifdef __HIP_PLATFORM_AMD__
        val += __shfl_down(val, offset);
#else
        val += __shfl_down_sync(0xffffffff, val, offset);
#endif
    return val;
}

inline __device__ double warpAllReduceSum_double(double val) {
#ifdef __HIP_PLATFORM_AMD__
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
#else
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
#endif
    return val;
}

__global__ void state_norm_squared_gpu(double* ret, GTYPE* state, ITYPE dim) {
    double sum = double(0.0);
    GTYPE tmp;
    ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (ITYPE i = idx; i < dim; i += blockDim.x * gridDim.x) {
        tmp = state[i];
        sum += tmp.x * tmp.x + tmp.y * tmp.y;
    }
    sum = warpReduceSum_double(sum);

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(ret, sum);
    }
}

__host__ double state_norm_squared_cublas_host(void* state, ITYPE dim) {
    gpublasStatus_t status;
    gpublasHandle_t handle;
    double norm;
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);

    /* Initialize CUBLAS */
    status = gpublasCreate(&handle);
    if (status != GPUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    status =
        gpublasDznrm2(handle, dim, (gpublasDoubleComplex*)state_gpu, 1, &norm);
    if (status != GPUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! cublasDznrm2 execution error.\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = gpublasDestroy(handle);

    if (status != GPUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error\n");
        return EXIT_FAILURE;
    }

    state = reinterpret_cast<void*>(state_gpu);
    return norm;
}

__host__ double state_norm_squared_host(
    void* state, ITYPE dim, void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    gpuError_t gpuStatus;
    double norm = 0.0;
    double* norm_gpu;
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);

    checkGpuErrors(
        gpuMalloc((void**)&norm_gpu, sizeof(double)), __FILE__, __LINE__);
    checkGpuErrors(gpuMemsetAsync(norm_gpu, 0, sizeof(double), *gpu_stream),
        __FILE__, __LINE__);

    ITYPE loop_dim = get_loop_dim_of_reduction_function(dim);
    unsigned int max_block_size =
        get_block_size_to_maximize_occupancy(state_norm_squared_gpu);
    unsigned int block = dim <= max_block_size ? dim : max_block_size;
    unsigned int grid = (dim + block - 1) / block;

    state_norm_squared_gpu<<<grid, block, 0, *gpu_stream>>>(
        norm_gpu, state_gpu, dim);

    // Check for any errors launching the kernel
    gpuStatus = gpuGetLastError();

    checkGpuErrors(gpuStatus, __FILE__, __LINE__);
    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(&norm, norm_gpu, sizeof(double),
                       gpuMemcpyDeviceToHost, *gpu_stream),
        __FILE__, __LINE__);

    checkGpuErrors(gpuFree(norm_gpu), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);
    return norm;
}

__global__ void measurement_distribution_entropy_gpu(
    double* ret, const GTYPE* state, ITYPE dim) {
    double sum = 0;
    const double eps = 1e-15;

    double prob;
    for (ITYPE index = blockIdx.x * blockDim.x + threadIdx.x; index < dim;
         index += blockDim.x * gridDim.x) {
        prob = gpuCabs(state[index]);
        prob = prob * prob;
        if (prob > eps) {
            sum += -1.0 * prob * log(prob);
        }
    }
    sum = warpReduceSum_double(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret[0]), sum);
    }
}

__host__ double measurement_distribution_entropy_host(
    void* state, ITYPE dim, void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    gpuError_t gpuStatus;
    double ent;
    double* ent_gpu;
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);

    checkGpuErrors(
        gpuMalloc((void**)&ent_gpu, sizeof(double)), __FILE__, __LINE__);
    checkGpuErrors(gpuMemsetAsync(ent_gpu, 0, sizeof(double), *gpu_stream),
        __FILE__, __LINE__);

    ITYPE loop_dim = get_loop_dim_of_reduction_function(dim);
    unsigned int max_block_size = get_block_size_to_maximize_occupancy(
        measurement_distribution_entropy_gpu);
    unsigned int block = dim <= max_block_size ? dim : max_block_size;
    unsigned int grid = (dim + block - 1) / block;

    measurement_distribution_entropy_gpu<<<grid, block, 0, *gpu_stream>>>(
        ent_gpu, state_gpu, dim);

    // Check for any errors launching the kernel
    gpuStatus = gpuGetLastError();

    checkGpuErrors(gpuStatus, __FILE__, __LINE__);
    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(&ent, ent_gpu, sizeof(double),
                       gpuMemcpyDeviceToHost, *gpu_stream),
        __FILE__, __LINE__);

    checkGpuErrors(gpuFree(ent_gpu), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);

    return ent;
}

__global__ void state_add_gpu(
    const GTYPE* state_added, GTYPE* state, ITYPE dim) {
    ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;

    // loop varaibles
    const ITYPE loop_dim = dim;
    if (state_index < loop_dim) {
        state[state_index] =
            gpuCadd(state[state_index], state_added[state_index]);
    }
}

__host__ void state_add_host(void* state_added, void* state, ITYPE dim,
    void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    GTYPE* state_added_gpu = reinterpret_cast<GTYPE*>(state_added);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);

    ITYPE loop_dim = dim;

    unsigned int max_block_size =
        get_block_size_to_maximize_occupancy(state_add_gpu);
    unsigned int block = dim <= max_block_size ? dim : max_block_size;
    unsigned int grid = (dim + block - 1) / block;

    state_add_gpu<<<grid, block, 0, *gpu_stream>>>(
        state_added_gpu, state_gpu, dim);

    checkGpuErrors(gpuGetLastError(), __FILE__, __LINE__);
    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);
    state_added = reinterpret_cast<void*>(state_added_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);
}

__global__ void state_multiply_gpu(const GTYPE coef, GTYPE* state, ITYPE dim) {
    ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;

    const ITYPE loop_dim = dim;
    if (state_index < loop_dim) {
        state[state_index] = gpuCmul(state[state_index], coef);
    }
}

__host__ void state_multiply_host(CPPCTYPE coef, void* state, ITYPE dim,
    void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    ITYPE loop_dim = dim;

    GTYPE coef_gpu = make_gpuDoubleComplex(coef.real(), coef.imag());
    unsigned int max_block_size =
        get_block_size_to_maximize_occupancy(state_multiply_gpu);
    unsigned int block = dim <= max_block_size ? dim : max_block_size;
    unsigned int grid = (dim + block - 1) / block;

    state_multiply_gpu<<<grid, block, 0, *gpu_stream>>>(
        coef_gpu, state_gpu, dim);

    checkGpuErrors(gpuGetLastError(), __FILE__, __LINE__);
    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);
}

__global__ void inner_product_gpu(
    GTYPE* ret, const GTYPE* psi, const GTYPE* phi, ITYPE dim) {
    GTYPE sum = make_gpuDoubleComplex(0.0, 0.0);
    for (ITYPE i = blockIdx.x * blockDim.x + threadIdx.x; i < dim;
         i += blockDim.x * gridDim.x) {
        sum = gpuCadd(sum, gpuCmul(gpuConj(psi[i]), phi[i]));
    }
    sum.x = warpReduceSum_double(sum.x);
    sum.y = warpReduceSum_double(sum.y);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret[0].x), sum.x);
        atomicAdd_double(&(ret[0].y), sum.y);
    }
}

/*
__host__ CPPCTYPE inner_product_cublas_host(const void *bra_state, const void
*ket_state, ITYPE dim) { const GTYPE* bra_state_gpu = reinterpret_cast<const
GTYPE*>(bra_state); const GTYPE* ket_state_gpu = reinterpret_cast<const
GTYPE*>(ket_state); cublasStatus_t status; cublasHandle_t handle; GTYPE ret_g;
    CPPCTYPE ret;

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

        status = cublasZdotc(handle, dim, bra_state_gpu, 1, ket_state_gpu, 1,
&ret_g); if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "!!!!
cublasZDotc execution error.\n"); return EXIT_FAILURE;
    }

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
*/

__host__ CPPCTYPE inner_product_cublas_host(const void* bra_state,
    const void* ket_state, ITYPE dim, void* stream,
    unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    const GTYPE* bra_state_gpu = reinterpret_cast<const GTYPE*>(bra_state);
    const GTYPE* ket_state_gpu = reinterpret_cast<const GTYPE*>(ket_state);
    gpublasStatus_t status;
    gpublasHandle_t handle;
    GTYPE ret_g;
    CPPCTYPE ret;

    /* Initialize CUBLAS */
    status = gpublasCreate(&handle);
    if (status != GPUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    status = gpublasSetStream(handle, *gpu_stream);
    if (status != GPUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! set cublas to cuda stream error\n");
        return EXIT_FAILURE;
    }

    status = gpublasZdotc(handle, dim, (gpublasDoubleComplex*)bra_state_gpu, 1,
        (gpublasDoubleComplex*)ket_state_gpu, 1, (gpublasDoubleComplex*)&ret_g);
    if (status != GPUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! cublasZDotc execution error.\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = gpublasDestroy(handle);

    if (status != GPUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error\n");
        return EXIT_FAILURE;
    }

    bra_state = reinterpret_cast<const void*>(bra_state_gpu);
    ket_state = reinterpret_cast<const void*>(ket_state_gpu);
    ret = CPPCTYPE(gpuCreal(ret_g), gpuCimag(ret_g));
    return ret;
}

__host__ CPPCTYPE inner_product_original_host(const void* bra_state,
    const void* ket_state, ITYPE dim, void* stream,
    unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    const GTYPE* bra_state_gpu = reinterpret_cast<const GTYPE*>(bra_state);
    const GTYPE* ket_state_gpu = reinterpret_cast<const GTYPE*>(ket_state);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    gpuError_t gpuStatus;
    CPPCTYPE ret = CPPCTYPE(0.0, 0.0);
    GTYPE* ret_gpu;

    checkGpuErrors(
        gpuMalloc((void**)&ret_gpu, sizeof(GTYPE)), __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(ret_gpu, &ret, sizeof(GTYPE),
                       gpuMemcpyHostToDevice, *gpu_stream),
        __FILE__, __LINE__);

    ITYPE loop_dim = get_loop_dim_of_reduction_function(dim);
    unsigned int max_block_size =
        get_block_size_to_maximize_occupancy(inner_product_gpu);
    unsigned int block = loop_dim <= max_block_size ? loop_dim : max_block_size;
    unsigned int grid = (loop_dim + block - 1) / block;

    inner_product_gpu<<<grid, block, 0, *gpu_stream>>>(
        ret_gpu, bra_state_gpu, ket_state_gpu, dim);

    // Check for any errors launching the kernel
    gpuStatus = gpuGetLastError();

    checkGpuErrors(gpuStatus, __FILE__, __LINE__);
    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(&ret, ret_gpu, sizeof(GTYPE),
                       gpuMemcpyDeviceToHost, *gpu_stream),
        __FILE__, __LINE__);

    checkGpuErrors(gpuFree(ret_gpu), __FILE__, __LINE__);
    bra_state = reinterpret_cast<const void*>(bra_state_gpu);
    ket_state = reinterpret_cast<const void*>(ket_state_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);
    return ret;
}

__host__ CPPCTYPE inner_product_host(const void* bra_state,
    const void* ket_state, ITYPE dim, void* stream,
    unsigned int device_number) {
    if (dim <= INT_MAX) {
        // あとでcublas版を使うように直す
        return inner_product_original_host(
            bra_state, ket_state, dim, stream, device_number);
        // return inner_product_cublas_host(bra_state, ket_state, dim, stream,
        // device_number);
    } else {
        return inner_product_original_host(
            bra_state, ket_state, dim, stream, device_number);
    }
}

__global__ void expectation_value_PauliI_gpu(
    double* ret, GTYPE* state, unsigned int target_qubit_index, ITYPE dim) {
    double sum = 0.0;
    ITYPE loop_dim = dim;
    GTYPE tmp_state;
    for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
         state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        tmp_state = state[state_index];
        sum += gpuCreal(gpuCmul(gpuConj(tmp_state), tmp_state));
    }
    sum = warpReduceSum_double(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret[0]), sum);
    }
}

__global__ void expectation_value_PauliX_gpu(
    double* ret, GTYPE* state, unsigned int target_qubit_index, ITYPE dim) {
    double sum = 0.0;
    ITYPE basis0, basis1;
    ITYPE loop_dim = dim >> 1;
    for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
         state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        basis0 = (state_index >> target_qubit_index);
        basis0 = basis0 << (target_qubit_index + 1);
        basis0 += state_index & ((1ULL << target_qubit_index) - 1);
        basis1 = basis0 ^ (1ULL << target_qubit_index);

        sum += gpuCreal(gpuCmul(gpuConj(state[basis0]), state[basis1]));
    }
    sum *= 2;
    sum = warpReduceSum_double(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret[0]), sum);
    }
}

__global__ void expectation_value_PauliY_gpu(
    double* ret, GTYPE* state, unsigned int target_qubit_index, ITYPE dim) {
    double sum = 0.0;
    ITYPE basis0, basis1;
    ITYPE loop_dim = dim >> 1;
    for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
         state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        basis0 = (state_index >> target_qubit_index);
        basis0 = basis0 << (target_qubit_index + 1);
        basis0 += state_index & ((1ULL << target_qubit_index) - 1);
        basis1 = basis0 ^ (1ULL << target_qubit_index);
        sum += gpuCimag(gpuCmul(gpuConj(state[basis0]), state[basis1]));
    }
    sum *= 2;
    sum = warpReduceSum_double(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret[0]), sum);
    }
}

__global__ void expectation_value_PauliZ_gpu(
    double* ret, GTYPE* state, unsigned int target_qubit_index, ITYPE dim) {
    double sum = 0.0;
    ITYPE basis0, basis1;
    ITYPE loop_dim = dim >> 1;
    for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
         state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        basis0 = (state_index >> target_qubit_index);
        basis0 = basis0 << (target_qubit_index + 1);
        basis0 += state_index & ((1ULL << target_qubit_index) - 1);
        basis1 = basis0 ^ (1ULL << target_qubit_index);
        sum += gpuCreal(gpuCmul(gpuConj(state[basis0]), state[basis0])) -
               gpuCreal(gpuCmul(gpuConj(state[basis1]), state[basis1]));
    }
    sum = warpReduceSum_double(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret[0]), sum);
    }
}

__host__ double expectation_value_single_qubit_Pauli_operator_host(
    unsigned int operator_index, unsigned int target_qubit_index, void* state,
    ITYPE dim, void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    double h_ret = 0.0;
    double* d_ret;

    checkGpuErrors(
        gpuMalloc((void**)&d_ret, sizeof(double)), __FILE__, __LINE__);
    checkGpuErrors(gpuMemsetAsync(d_ret, 0, sizeof(double), *gpu_stream),
        __FILE__, __LINE__);

    if (operator_index == 1) {
        ITYPE loop_dim = get_loop_dim_of_reduction_function(dim >> 1);
        unsigned int max_block_size =
            get_block_size_to_maximize_occupancy(expectation_value_PauliX_gpu);
        unsigned int block =
            loop_dim <= max_block_size ? loop_dim : max_block_size;
        unsigned int grid = (loop_dim + block - 1) / block;
        expectation_value_PauliX_gpu<<<grid, block, 0, *gpu_stream>>>(
            d_ret, state_gpu, target_qubit_index, dim);
    } else if (operator_index == 2) {
        ITYPE loop_dim = get_loop_dim_of_reduction_function(dim >> 1);
        unsigned int max_block_size =
            get_block_size_to_maximize_occupancy(expectation_value_PauliY_gpu);
        unsigned int block =
            loop_dim <= max_block_size ? loop_dim : max_block_size;
        unsigned int grid = (loop_dim + block - 1) / block;
        expectation_value_PauliY_gpu<<<grid, block, 0, *gpu_stream>>>(
            d_ret, state_gpu, target_qubit_index, dim);
    } else if (operator_index == 3) {
        ITYPE loop_dim = get_loop_dim_of_reduction_function(dim >> 1);
        unsigned int max_block_size =
            get_block_size_to_maximize_occupancy(expectation_value_PauliZ_gpu);
        unsigned int block =
            loop_dim <= max_block_size ? loop_dim : max_block_size;
        unsigned int grid = (loop_dim + block - 1) / block;
        expectation_value_PauliZ_gpu<<<grid, block, 0, *gpu_stream>>>(
            d_ret, state_gpu, target_qubit_index, dim);
    } else if (operator_index == 0) {
        ITYPE loop_dim = get_loop_dim_of_reduction_function(dim);
        unsigned int max_block_size =
            get_block_size_to_maximize_occupancy(expectation_value_PauliI_gpu);
        unsigned int block =
            loop_dim <= max_block_size ? loop_dim : max_block_size;
        unsigned int grid = (loop_dim + block - 1) / block;
        expectation_value_PauliI_gpu<<<grid, block, 0, *gpu_stream>>>(
            d_ret, state_gpu, target_qubit_index, dim);
    } else {
        printf("operator_index must be an integer of 0, 1, 2, or 3!!");
    }

    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(&h_ret, d_ret, sizeof(double),
                       gpuMemcpyDeviceToHost, *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(gpuFree(d_ret), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);
    return h_ret;
}

__device__ void multi_Z_gate_device(ITYPE bit_mask, ITYPE DIM, GTYPE* psi_gpu) {
    ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int minus_cnt = 0;
    if (idx < DIM) {
        minus_cnt = __popcll(idx & bit_mask);
        if (minus_cnt & 1)
            psi_gpu[idx] =
                make_gpuDoubleComplex(-psi_gpu[idx].x, -psi_gpu[idx].y);
    }
}

__global__ void multi_Z_gate_gpu(ITYPE bit_mask, ITYPE DIM, GTYPE* psi_gpu) {
    multi_Z_gate_device(bit_mask, DIM, psi_gpu);
}

__host__ void multi_Z_gate_host(int* gates, void* state, ITYPE dim,
    int n_qubits, void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    ITYPE bit_mask = 0;
    for (int i = 0; i < n_qubits; ++i) {
        if (gates[i] == 3) bit_mask ^= (1ULL << i);
    }
    gpuError_t gpuStatus;
    unsigned int max_block_size =
        get_block_size_to_maximize_occupancy(multi_Z_gate_gpu);
    unsigned int block = dim <= max_block_size ? dim : max_block_size;
    unsigned int grid = (dim + block - 1) / block;
    multi_Z_gate_gpu<<<grid, block, 0, *gpu_stream>>>(bit_mask, dim, state_gpu);
    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    gpuStatus = gpuGetLastError();
    checkGpuErrors(gpuStatus, __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);
}

__device__ GTYPE multi_Z_get_expectation_value_device(
    ITYPE idx, ITYPE bit_mask, ITYPE dim, GTYPE* psi_gpu) {
    GTYPE ret = make_gpuDoubleComplex(0.0, 0.0);
    // ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int minus_cnt = 0;
    if (idx < dim) {
        GTYPE tmp_psi_gpu = psi_gpu[idx];
        minus_cnt = __popcll(idx & bit_mask);
        if (minus_cnt & 1)
            tmp_psi_gpu = make_gpuDoubleComplex(-tmp_psi_gpu.x, -tmp_psi_gpu.y);
        ret = gpuCmul(gpuConj(psi_gpu[idx]), tmp_psi_gpu);
    }
    return ret;
}

__global__ void multi_Z_get_expectation_value_gpu(
    GTYPE* ret, ITYPE bit_mask, ITYPE DIM, GTYPE* psi_gpu) {
    GTYPE sum = make_gpuDoubleComplex(0.0, 0.0);
    GTYPE tmp;
    for (ITYPE state = blockIdx.x * blockDim.x + threadIdx.x; state < DIM;
         state += blockDim.x * gridDim.x) {
        tmp =
            multi_Z_get_expectation_value_device(state, bit_mask, DIM, psi_gpu);
        sum = gpuCadd(sum, tmp);
    }
    sum.x = warpReduceSum_double(sum.x);
    sum.y = warpReduceSum_double(sum.y);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret[0].x), sum.x);
        atomicAdd_double(&(ret[0].y), sum.y);
    }
}

__device__ GTYPE multipauli_get_expectation_value_device(
    ITYPE idx, ITYPE DIM, GTYPE* psi_gpu, int n_qubits) {
    GTYPE ret;
    GTYPE tmp_psi, tmp_prev_state_psi, tmp_state_psi;
    ITYPE prev_state, state;
    int num_y1 = 0;
    int num_z1 = 0;
    int i_cnt = 0;
    int minus_cnt = 0;
    if (idx < DIM) {
        i_cnt = num_pauli_op_gpu[2];
        num_y1 = __popcll(idx & bit_mask_gpu[2]);
        num_z1 = __popcll(idx & bit_mask_gpu[3]);
        minus_cnt = num_y1 + num_z1;
        prev_state = idx;
        state = idx ^ (bit_mask_gpu[1] + bit_mask_gpu[2]);
        tmp_prev_state_psi = psi_gpu[prev_state];
        tmp_state_psi = psi_gpu[state];
        // swap
        tmp_psi = tmp_state_psi;
        tmp_state_psi = tmp_prev_state_psi;
        tmp_prev_state_psi = tmp_psi;
        if (minus_cnt & 1)
            tmp_state_psi =
                make_gpuDoubleComplex(-tmp_state_psi.x, -tmp_state_psi.y);
        if (i_cnt & 1)
            tmp_state_psi =
                make_gpuDoubleComplex(tmp_state_psi.y, tmp_state_psi.x);
        if ((i_cnt >> 1) & 1)
            tmp_state_psi =
                make_gpuDoubleComplex(-tmp_state_psi.x, -tmp_state_psi.y);
        // tmp_state      -> state      : state*conj(tmp_state)
        // tmp_prev_state -> prev_state : prev_state*conj(tmp_prev_state)
        ret = gpuCmul(tmp_state_psi, gpuConj(psi_gpu[state]));
    }
    return ret;
}

__global__ void multipauli_get_expectation_value_gpu(
    GTYPE* ret, ITYPE DIM, GTYPE* psi_gpu, int n_qubits) {
    GTYPE sum = make_gpuDoubleComplex(0.0, 0.0);
    GTYPE tmp;
    for (ITYPE state = blockIdx.x * blockDim.x + threadIdx.x; state < DIM;
         state += blockDim.x * gridDim.x) {
        tmp = multipauli_get_expectation_value_device(
            state, DIM, psi_gpu, n_qubits);
        sum = gpuCadd(sum, tmp);
    }
    sum.x = warpReduceSum_double(sum.x);
    sum.y = warpReduceSum_double(sum.y);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret[0].x), sum.x);
        atomicAdd_double(&(ret[0].y), sum.y);
    }
}

__host__ double multipauli_get_expectation_value_host(unsigned int* gates,
    void* state, ITYPE dim, int n_qubits, void* stream,
    unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    CPPCTYPE ret[1];
    ret[0] = CPPCTYPE(0, 0);
    GTYPE* ret_gpu;

    checkGpuErrors(
        gpuMalloc((void**)&ret_gpu, sizeof(GTYPE)), __FILE__, __LINE__);
    checkGpuErrors(gpuMemsetAsync(ret_gpu, 0, sizeof(double), *gpu_stream),
        __FILE__, __LINE__);

    ITYPE loop_dim = get_loop_dim_of_reduction_function(dim);

    unsigned int num_pauli_op[4] = {0, 0, 0, 0};
    for (int i = 0; i < n_qubits; ++i) ++num_pauli_op[gates[i]];
    ITYPE bit_mask[4] = {0, 0, 0, 0};
    for (int i = 0; i < n_qubits; ++i) {
        bit_mask[gates[i]] ^= (1ULL << i);
    }
    if (num_pauli_op[1] == 0 && num_pauli_op[2] == 0) {
        unsigned int max_block_size = get_block_size_to_maximize_occupancy(
            multi_Z_get_expectation_value_gpu);
        unsigned int block =
            loop_dim <= max_block_size ? loop_dim : max_block_size;
        unsigned int grid = (loop_dim + block - 1) / block;
        multi_Z_get_expectation_value_gpu<<<grid, block, 0, *gpu_stream>>>(
            ret_gpu, bit_mask[3], dim, state_gpu);
        checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
        checkGpuErrors(gpuGetLastError(), __FILE__, __LINE__);
        checkGpuErrors(gpuMemcpyAsync(ret, ret_gpu, sizeof(CPPCTYPE),
                           gpuMemcpyDeviceToHost, *gpu_stream),
            __FILE__, __LINE__);
        checkGpuErrors(gpuFree(ret_gpu), __FILE__, __LINE__);
        state = reinterpret_cast<void*>(state_gpu);
        return ret[0].real();
    }

    unsigned int max_block_size = get_block_size_to_maximize_occupancy(
        multipauli_get_expectation_value_gpu);
    unsigned int block = loop_dim <= max_block_size ? loop_dim : max_block_size;
    unsigned int grid = (loop_dim + block - 1) / block;
    checkGpuErrors(
        gpuMemcpyToSymbolAsync(GPU_SYMBOL(num_pauli_op_gpu), num_pauli_op,
            sizeof(unsigned int) * 4, 0, gpuMemcpyHostToDevice, *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(
        gpuMemcpyToSymbolAsync(GPU_SYMBOL(bit_mask_gpu), bit_mask,
            sizeof(ITYPE) * 4, 0, gpuMemcpyHostToDevice, *gpu_stream),
        __FILE__, __LINE__);

    multipauli_get_expectation_value_gpu<<<grid, block, 0, *gpu_stream>>>(
        ret_gpu, dim, state_gpu, n_qubits);

    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    checkGpuErrors(gpuGetLastError(), __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(ret, ret_gpu, sizeof(CPPCTYPE),
                       gpuMemcpyDeviceToHost, *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(gpuFree(ret_gpu), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);
    return ret[0].real();
}

// calculate probability with which we obtain 0 at target qubit
__global__ void M0_prob_gpu(
    double* ret, UINT target_qubit_index, const GTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim >> 1;
    double sum = 0.;
    double tmp;
    for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
         state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        ITYPE basis_0 =
            insert_zero_to_basis_index_device(state_index, target_qubit_index);
        tmp = gpuCabs(state[basis_0]);
        sum += tmp * tmp;
    }
    sum = warpReduceSum_double(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret[0]), sum);
    }
}

// calculate probability with which we obtain 0 at target qubit
__host__ double M0_prob_host(UINT target_qubit_index, void* state, ITYPE dim,
    void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    double ret[1] = {0.0};
    double* ret_gpu;

    checkGpuErrors(
        gpuMalloc((void**)&ret_gpu, sizeof(double)), __FILE__, __LINE__);
    checkGpuErrors(gpuMemsetAsync(ret_gpu, 0, sizeof(double), *gpu_stream),
        __FILE__, __LINE__);

    ITYPE loop_dim = get_loop_dim_of_reduction_function(dim >> 1);

    unsigned int max_block_size =
        get_block_size_to_maximize_occupancy(M0_prob_gpu);
    unsigned int block = loop_dim <= max_block_size ? loop_dim : max_block_size;
    unsigned int grid = (loop_dim + block - 1) / block;

    M0_prob_gpu<<<grid, block, 0, *gpu_stream>>>(
        ret_gpu, target_qubit_index, state_gpu, dim);

    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    checkGpuErrors(gpuGetLastError(), __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(ret, ret_gpu, sizeof(double),
                       gpuMemcpyDeviceToHost, *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(gpuFree(ret_gpu), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);
    return ret[0];
}

// calculate probability with which we obtain 1 at target qubit
__global__ void M1_prob_gpu(
    double* ret, UINT target_qubit_index, const GTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim >> 1;
    const ITYPE mask = 1ULL << target_qubit_index;
    double sum = 0.;
    double tmp;
    for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
         state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        ITYPE basis_1 =
            insert_zero_to_basis_index_device(state_index, target_qubit_index) ^
            mask;
        tmp = gpuCabs(state[basis_1]);
        sum += tmp * tmp;
    }
    sum = warpReduceSum_double(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret[0]), sum);
    }
}

__host__ double M1_prob_host(UINT target_qubit_index, void* state, ITYPE dim,
    void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    double ret[1] = {0.0};
    double* ret_gpu;

    checkGpuErrors(
        gpuMalloc((void**)&ret_gpu, sizeof(double)), __FILE__, __LINE__);
    checkGpuErrors(gpuMemsetAsync(ret_gpu, 0, sizeof(double), *gpu_stream),
        __FILE__, __LINE__);

    ITYPE loop_dim = get_loop_dim_of_reduction_function(dim >> 1);
    unsigned int max_block_size =
        get_block_size_to_maximize_occupancy(M1_prob_gpu);
    unsigned int block = loop_dim <= max_block_size ? loop_dim : max_block_size;
    unsigned int grid = (loop_dim + block - 1) / block;

    M1_prob_gpu<<<grid, block, 0, *gpu_stream>>>(
        ret_gpu, target_qubit_index, state_gpu, dim);

    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    checkGpuErrors(gpuGetLastError(), __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(ret, ret_gpu, sizeof(double),
                       gpuMemcpyDeviceToHost, *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(gpuFree(ret_gpu), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);
    return ret[0];
}

// calculate merginal probability with which we obtain the set of values
// measured_value_list at sorted_target_qubit_index_list warning:
// sorted_target_qubit_index_list must be sorted.
__global__ void marginal_prob_gpu(double* ret_gpu,
    const UINT* sorted_target_qubit_index_list, const UINT* measured_value_list,
    UINT target_qubit_index_count, const GTYPE* state, ITYPE dim) {
    ITYPE loop_dim = dim >> target_qubit_index_count;
    double sum = 0.;
    double tmp;
    for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
         state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        ITYPE basis = state_index;
        for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
            UINT insert_index = sorted_target_qubit_index_list[cursor];
            ITYPE mask = 1ULL << insert_index;
            basis = insert_zero_to_basis_index_device(basis, insert_index);
            basis ^= mask * measured_value_list[cursor];
        }
        tmp = gpuCabs(state[basis]);
        sum += tmp * tmp;
    }
    sum = warpReduceSum_double(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret_gpu[0]), sum);
    }
}

__host__ double marginal_prob_host(UINT* sorted_target_qubit_index_list,
    UINT* measured_value_list, UINT target_qubit_index_count, void* state,
    ITYPE dim, void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    double ret[1] = {0.0};
    double* ret_gpu;
    UINT* sorted_target_qubit_index_list_gpu;
    UINT* measured_value_list_gpu;

    checkGpuErrors(
        gpuMalloc((void**)&ret_gpu, sizeof(double)), __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(ret_gpu, ret, sizeof(double),
                       gpuMemcpyHostToDevice, *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(gpuMalloc((void**)&sorted_target_qubit_index_list_gpu,
                       sizeof(UINT) * target_qubit_index_count),
        __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(sorted_target_qubit_index_list_gpu,
                       sorted_target_qubit_index_list,
                       sizeof(UINT) * target_qubit_index_count,
                       gpuMemcpyHostToDevice, *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(gpuMalloc((void**)&measured_value_list_gpu,
                       sizeof(UINT) * target_qubit_index_count),
        __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(measured_value_list_gpu, measured_value_list,
                       sizeof(UINT) * target_qubit_index_count,
                       gpuMemcpyHostToDevice, *gpu_stream),
        __FILE__, __LINE__);

    ITYPE loop_dim =
        get_loop_dim_of_reduction_function(dim >> target_qubit_index_count);
    unsigned int max_block_size =
        get_block_size_to_maximize_occupancy(marginal_prob_gpu);
    unsigned int block = loop_dim <= max_block_size ? loop_dim : max_block_size;
    unsigned int grid = (loop_dim + block - 1) / block;

    marginal_prob_gpu<<<grid, block, 0, *gpu_stream>>>(ret_gpu,
        sorted_target_qubit_index_list_gpu, measured_value_list_gpu,
        target_qubit_index_count, state_gpu, dim);

    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    checkGpuErrors(gpuGetLastError(), __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(ret, ret_gpu, sizeof(double),
                       gpuMemcpyDeviceToHost, *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(gpuFree(ret_gpu), __FILE__, __LINE__);
    checkGpuErrors(
        gpuFree(sorted_target_qubit_index_list_gpu), __FILE__, __LINE__);
    checkGpuErrors(gpuFree(measured_value_list_gpu), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);
    return ret[0];
}

__global__ void expectation_value_multi_qubit_Pauli_operator_XZ_mask_gpu(
    double* ret_gpu, ITYPE bit_flip_mask, ITYPE phase_flip_mask,
    UINT global_phase_90rot_count, UINT pivot_qubit_index, GTYPE* state,
    ITYPE dim) {
    const ITYPE loop_dim = dim >> 1;
    double sum = 0.;
    double tmp;
    for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
         state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        ITYPE basis_0 =
            insert_zero_to_basis_index_device(state_index, pivot_qubit_index);
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;
        UINT sign_0 = __popcll(basis_0 & phase_flip_mask) & 1;

        tmp =
            gpuCreal(gpuCmul(gpuCmul(state[basis_0], gpuConj(state[basis_1])),
                PHASE_90ROT_gpu[(global_phase_90rot_count + sign_0 * 2) & 3])) *
            2.0;
        sum += tmp;
    }
    sum = warpReduceSum_double(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret_gpu[0]), sum);
    }
}

__host__ double expectation_value_multi_qubit_Pauli_operator_XZ_mask_host(
    ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, void* state, ITYPE dim, void* stream,
    unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    gpuError_t gpuStatus;
    double ret;
    double* ret_gpu;
    CPPCTYPE PHASE_90ROT[4] = {CPPCTYPE(1.0, 0.0), CPPCTYPE(0.0, 1.0),
        CPPCTYPE(-1.0, 0.0), CPPCTYPE(0.0, -1.0)};

    checkGpuErrors(
        gpuMalloc((void**)&ret_gpu, sizeof(double)), __FILE__, __LINE__);
    checkGpuErrors(gpuMemsetAsync(ret_gpu, 0, sizeof(double), *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(
        gpuMemcpyToSymbolAsync(GPU_SYMBOL(PHASE_90ROT_gpu), PHASE_90ROT,
            sizeof(GTYPE) * 4, 0, gpuMemcpyHostToDevice, *gpu_stream),
        __FILE__, __LINE__);

    ITYPE loop_dim = get_loop_dim_of_reduction_function(dim >> 1);
    unsigned int max_block_size = get_block_size_to_maximize_occupancy(
        expectation_value_multi_qubit_Pauli_operator_XZ_mask_gpu);
    unsigned int block = loop_dim <= max_block_size ? loop_dim : max_block_size;
    unsigned int grid = (loop_dim + block - 1) / block;

    expectation_value_multi_qubit_Pauli_operator_XZ_mask_gpu<<<grid, block, 0,
        *gpu_stream>>>(ret_gpu, bit_flip_mask, phase_flip_mask,
        global_phase_90rot_count, pivot_qubit_index, state_gpu, dim);

    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    gpuStatus = gpuGetLastError();
    checkGpuErrors(gpuStatus, __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(&ret, ret_gpu, sizeof(double),
                       gpuMemcpyDeviceToHost, *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(gpuFree(ret_gpu), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);

    return ret;
}

__global__ void expectation_value_multi_qubit_Pauli_operator_Z_mask_gpu(
    double* ret_gpu, ITYPE phase_flip_mask, const GTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim;
    double sum = 0.;
    double tmp;
    for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
         state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        UINT bit_parity = __popcll(state_index & phase_flip_mask) & 1;
        int sign = 1 - 2 * bit_parity;
        tmp = gpuCabs(state[state_index]);
        sum += tmp * tmp * sign;
    }
    sum = warpReduceSum_double(sum);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret_gpu[0]), sum);
    }
}

__host__ double expectation_value_multi_qubit_Pauli_operator_Z_mask_host(
    ITYPE phase_flip_mask, void* state, ITYPE dim, void* stream,
    unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    gpuError_t gpuStatus;
    double ret;
    double* ret_gpu;

    checkGpuErrors(
        gpuMalloc((void**)&ret_gpu, sizeof(double)), __FILE__, __LINE__);
    checkGpuErrors(gpuMemsetAsync(ret_gpu, 0, sizeof(double), *gpu_stream),
        __FILE__, __LINE__);

    ITYPE loop_dim = get_loop_dim_of_reduction_function(dim);
    unsigned int max_block_size = get_block_size_to_maximize_occupancy(
        expectation_value_multi_qubit_Pauli_operator_Z_mask_gpu);
    unsigned int block = loop_dim <= max_block_size ? loop_dim : max_block_size;
    unsigned int grid = (loop_dim + block - 1) / block;

    expectation_value_multi_qubit_Pauli_operator_Z_mask_gpu<<<grid, block, 0,
        *gpu_stream>>>(ret_gpu, phase_flip_mask, state_gpu, dim);

    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    gpuStatus = gpuGetLastError();
    checkGpuErrors(gpuStatus, __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(&ret, ret_gpu, sizeof(double),
                       gpuMemcpyDeviceToHost, *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(gpuFree(ret_gpu), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);

    return ret;
}

__host__ double expectation_value_multi_qubit_Pauli_operator_partial_list_host(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, void* state, ITYPE dim, void* stream,
    unsigned int device_number) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list_gsim(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    double result;
    if (bit_flip_mask == 0) {
        result = expectation_value_multi_qubit_Pauli_operator_Z_mask_host(
            phase_flip_mask, state, dim, stream, device_number);
    } else {
        result = expectation_value_multi_qubit_Pauli_operator_XZ_mask_host(
            bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
            pivot_qubit_index, state, dim, stream, device_number);
    }
    return result;
}

__host__ double expectation_value_multi_qubit_Pauli_operator_whole_list_host(
    const UINT* Pauli_operator_type_list, UINT qubit_count, void* state,
    ITYPE dim, void* stream, unsigned int device_number) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list_gsim(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count,
        &pivot_qubit_index);
    double result;
    if (bit_flip_mask == 0) {
        result = expectation_value_multi_qubit_Pauli_operator_Z_mask_host(
            phase_flip_mask, state, dim, stream, device_number);
    } else {
        result = expectation_value_multi_qubit_Pauli_operator_XZ_mask_host(
            bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
            pivot_qubit_index, state, dim, stream, device_number);
    }
    return result;
}

__global__ void transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_gpu(
    GTYPE* ret_gpu, ITYPE bit_flip_mask, ITYPE phase_flip_mask,
    UINT global_phase_90rot_count, UINT pivot_qubit_index,
    const GTYPE* state_bra, const GTYPE* state_ket, ITYPE dim) {
    const ITYPE loop_dim = dim >> 1;

    GTYPE sum = make_gpuDoubleComplex(0.0, 0.0);
    GTYPE tmp;
    for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
         state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        ITYPE basis_0 =
            insert_zero_to_basis_index_device(state_index, pivot_qubit_index);
        ITYPE basis_1 = basis_0 ^ bit_flip_mask;

        UINT sign_0 = __popcll(basis_0 & phase_flip_mask) & 1;
        tmp = gpuCmul(gpuCmul(state_ket[basis_0], gpuConj(state_bra[basis_1])),
            PHASE_90ROT_gpu[(global_phase_90rot_count + sign_0 * 2) & 3]);
        sum = gpuCadd(sum, tmp);

        UINT sign_1 = __popcll(basis_1 & phase_flip_mask) & 1;
        tmp = gpuCmul(gpuCmul(state_ket[basis_1], gpuConj(state_bra[basis_0])),
            PHASE_90ROT_gpu[(global_phase_90rot_count + sign_1 * 2) & 3]);
        sum = gpuCadd(sum, tmp);
    }
    sum.x = warpReduceSum_double(sum.x);
    sum.y = warpReduceSum_double(sum.y);
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret_gpu[0].x), sum.x);
        atomicAdd_double(&(ret_gpu[0].y), sum.y);
    }
}

__host__ CPPCTYPE transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_host(
    ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, void* state_bra, void* state_ket, ITYPE dim,
    void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    GTYPE* state_bra_gpu = reinterpret_cast<GTYPE*>(state_bra);
    GTYPE* state_ket_gpu = reinterpret_cast<GTYPE*>(state_ket);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    CPPCTYPE ret;
    GTYPE* ret_gpu;

    checkGpuErrors(
        gpuMalloc((void**)&ret_gpu, sizeof(GTYPE)), __FILE__, __LINE__);
    checkGpuErrors(gpuMemsetAsync(ret_gpu, 0, sizeof(GTYPE), *gpu_stream),
        __FILE__, __LINE__);

    ITYPE loop_dim = get_loop_dim_of_reduction_function(dim >> 1);
    unsigned int max_block_size = get_block_size_to_maximize_occupancy(
        transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_gpu);
    unsigned int block = loop_dim <= max_block_size ? loop_dim : max_block_size;
    unsigned int grid = (loop_dim + block - 1) / block;

    transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_gpu<<<grid, block,
        0, *gpu_stream>>>(ret_gpu, bit_flip_mask, phase_flip_mask,
        global_phase_90rot_count, pivot_qubit_index, state_bra_gpu,
        state_ket_gpu, dim);

    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    checkGpuErrors(gpuGetLastError(), __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(&ret, ret_gpu, sizeof(GTYPE),
                       gpuMemcpyDeviceToHost, *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(gpuFree(ret_gpu), __FILE__, __LINE__);
    state_bra = reinterpret_cast<void*>(state_bra_gpu);
    state_ket = reinterpret_cast<void*>(state_ket_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);
    return ret;
}

__global__ void transition_amplitude_multi_qubit_Pauli_operator_Z_mask_gpu(
    GTYPE* ret, ITYPE phase_flip_mask, GTYPE* state_bra, GTYPE* state_ket,
    ITYPE dim) {
    const ITYPE loop_dim = dim;
    GTYPE sum = make_gpuDoubleComplex(0.0, 0.0);
    for (ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
         state_index < loop_dim; state_index += blockDim.x * gridDim.x) {
        int bit_parity = __popcll(state_index & phase_flip_mask) & 1;
        int sign = 1 - 2 * bit_parity;
        GTYPE tmp =
            gpuCmul(state_ket[state_index], gpuConj(state_bra[state_index]));
        tmp = gpuCmul(make_gpuDoubleComplex((double)sign, 0.0), tmp);
        sum = gpuCadd(sum, tmp);
    }
    sum.x = warpReduceSum_double(sum.x);
    sum.y = warpReduceSum_double(sum.y);

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd_double(&(ret[0].x), sum.x);
        atomicAdd_double(&(ret[0].y), sum.y);
    }
}

__host__ CPPCTYPE transition_amplitude_multi_qubit_Pauli_operator_Z_mask_host(
    ITYPE phase_flip_mask, void* state_bra, void* state_ket, ITYPE dim,
    void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    gpuError_t gpuStatus;
    GTYPE* state_bra_gpu = reinterpret_cast<GTYPE*>(state_bra);
    GTYPE* state_ket_gpu = reinterpret_cast<GTYPE*>(state_ket);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    CPPCTYPE ret;
    GTYPE* ret_gpu;

    checkGpuErrors(
        gpuMalloc((void**)&ret_gpu, sizeof(GTYPE)), __FILE__, __LINE__);
    checkGpuErrors(gpuMemsetAsync(ret_gpu, 0, sizeof(GTYPE), *gpu_stream),
        __FILE__, __LINE__);

    ITYPE loop_dim = get_loop_dim_of_reduction_function(dim);
    unsigned int max_block_size = get_block_size_to_maximize_occupancy(
        transition_amplitude_multi_qubit_Pauli_operator_Z_mask_gpu);
    unsigned int block = loop_dim <= max_block_size ? loop_dim : max_block_size;
    unsigned int grid = (loop_dim + block - 1) / block;

    transition_amplitude_multi_qubit_Pauli_operator_Z_mask_gpu<<<grid, block, 0,
        *gpu_stream>>>(
        ret_gpu, phase_flip_mask, state_bra_gpu, state_ket_gpu, dim);

    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    gpuStatus = gpuGetLastError();
    checkGpuErrors(gpuStatus, __FILE__, __LINE__);
    checkGpuErrors(gpuMemcpyAsync(&ret, ret_gpu, sizeof(GTYPE),
                       gpuMemcpyDeviceToHost, *gpu_stream),
        __FILE__, __LINE__);
    checkGpuErrors(gpuFree(ret_gpu), __FILE__, __LINE__);
    state_bra = reinterpret_cast<void*>(state_bra_gpu);
    state_ket = reinterpret_cast<void*>(state_ket_gpu);
    // stream = reinterpret_cast<void*>(cuda_stream);
    return ret;
}

__host__ CPPCTYPE
transition_amplitude_multi_qubit_Pauli_operator_partial_list_host(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, void* state_bra, void* state_ket, ITYPE dim,
    void* stream, unsigned int device_number) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list_gsim(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    CPPCTYPE result;
    if (bit_flip_mask == 0) {
        result = transition_amplitude_multi_qubit_Pauli_operator_Z_mask_host(
            phase_flip_mask, state_bra, state_ket, dim, stream, device_number);
    } else {
        result = transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_host(
            bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
            pivot_qubit_index, state_bra, state_ket, dim, stream,
            device_number);
    }
    return result;
}

__host__ CPPCTYPE
transition_amplitude_multi_qubit_Pauli_operator_whole_list_host(
    const UINT* Pauli_operator_type_list, UINT qubit_count, void* state_bra,
    void* state_ket, ITYPE dim, void* stream, unsigned int device_number) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list_gsim(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count,
        &pivot_qubit_index);
    CPPCTYPE result;
    if (bit_flip_mask == 0) {
        result = transition_amplitude_multi_qubit_Pauli_operator_Z_mask_host(
            phase_flip_mask, state_bra, state_ket, dim, stream, device_number);
    } else {
        result = transition_amplitude_multi_qubit_Pauli_operator_XZ_mask_host(
            bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
            pivot_qubit_index, state_bra, state_ket, dim, stream,
            device_number);
    }
    return result;
}
