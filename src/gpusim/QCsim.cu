#include "device_launch_parameters.h"
#include <cuda_runtime.h>
//#include <cuda.h>

#ifdef __cplusplus
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <complex>
//#include <sys/time.h>
#else
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#endif

#include <cuComplex.h>
#include "util.h"
#include "util.cuh"
#include "util_common.h"
#include "update_ops_cuda.h"

__host__ void U1_gate_host(double lambda, unsigned int target_qubit_index, void *state, ITYPE DIM){
	CTYPE U_gate[4];

	U_gate[0] = std::complex<double>(1.0, 0.0); // make_cuDoubleComplex(1.0, 0.0);
	U_gate[1] = std::complex<double>(0.0, 0.0); //make_cuDoubleComplex(0.0, 0.0);
	U_gate[2] = std::complex<double>(0.0, 0.0); // make_cuDoubleComplex(0.0, 0.0);
	U_gate[3] = std::complex<double>(cos(lambda), sin(lambda)); // make_cuDoubleComplex(cos(lambda), sin(lambda));

    single_qubit_dense_matrix_gate_host(target_qubit_index, U_gate, state, DIM);

}

__host__ void U2_gate_host(double lambda, double phi, unsigned int target_qubit_index, void *state, ITYPE DIM){
	CTYPE U_gate[4];
	double sqrt2_inv = 1.0 / sqrt(2.0);
	CTYPE exp_val1 = CTYPE(cos(phi), sin(phi));
	CTYPE exp_val2 = CTYPE(cos(lambda), sin(lambda));

	U_gate[0] = CTYPE(sqrt2_inv, 0.0);
	U_gate[1] = CTYPE(-cos(lambda) / sqrt(2.0), -sin(lambda) / sqrt(2.0));
	U_gate[2] = CTYPE(cos(phi) / sqrt(2.0), sin(phi) / sqrt(2.0));
	U_gate[3] = exp_val1 * exp_val2;
	U_gate[3] = CTYPE(U_gate[3].real() / sqrt(2.0), U_gate[3].imag() / sqrt(2.0));

    single_qubit_dense_matrix_gate_host(target_qubit_index, U_gate, state, DIM);
	
}

__host__ void U3_gate_host(double lambda, double phi, double theta, unsigned int target_qubit_index, void* state, ITYPE DIM){
	CTYPE U_gate[4];
	double sqrt2_inv = 1.0 / sqrt(2.0);
	CTYPE exp_val1 = CTYPE(cos(phi), sin(phi));
	CTYPE exp_val2 = CTYPE(cos(lambda), sin(lambda));
	double cos_val = cos(theta / 2);
	double sin_val = sin(theta / 2);

	U_gate[0] = CTYPE(cos_val, 0.0);
	U_gate[1] = CTYPE(-cos(lambda)*sin_val, -sin(lambda)*sin_val);
	U_gate[2] = CTYPE(cos(phi)*sin_val, sin(phi)*sin_val);
	U_gate[3] = exp_val1 * exp_val2;
	U_gate[3] = CTYPE(U_gate[3].real()*cos_val, U_gate[3].imag()*cos_val);

    single_qubit_dense_matrix_gate_host(target_qubit_index, U_gate, state, DIM);

}

__global__ void multi_Pauli_gate_gpu(
	int* gates, ITYPE bit_mask_XY, int* num_pauli_op, ITYPE DIM, GTYPE *psi_gpu, int n_qubits
	){
	ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE IZ_state, XY_state, prev_state;
	int target_qubit_index, IZ_itr, XY_itr;
	GTYPE tmp_psi;
	ITYPE state = 0;
	int num_y1 = 0;
	int num_z1 = 0;
	int i_cnt = 0;
	int minus_cnt = 0;
	if (idx < (DIM >> 1)){
		IZ_state = idx & ((1 << (num_pauli_op[0] + num_pauli_op[3])) - 1);
		XY_state = idx >> (num_pauli_op[0] + num_pauli_op[3]);
		IZ_itr = (num_pauli_op[0] + num_pauli_op[3]) - 1;
		XY_itr = (num_pauli_op[1] + num_pauli_op[2]) - 1;
		for (int i = 0; i < n_qubits; ++i){
			target_qubit_index = n_qubits - 1 - i;
			switch (gates[target_qubit_index]){
			case 0:
				if ((IZ_state >> IZ_itr) & 1) state += (1LL << target_qubit_index);
				--IZ_itr;
				break;
			case 1:
				if ((XY_state >> XY_itr) & 1) state += (1LL << target_qubit_index);
				--XY_itr;
				break;
			case 2:
				if ((XY_state >> XY_itr) & 1){
					++minus_cnt;
					++num_y1;
					state += (1LL << target_qubit_index);
				}
				--XY_itr;
				++i_cnt;
				break;
			case 3:
				if ((IZ_state >> IZ_itr) & 1){
					++minus_cnt;
					++num_z1;
					state += (1LL << target_qubit_index);
				}
				--IZ_itr;
				break;
			}
		}
		prev_state = state;
		state = state^bit_mask_XY;
		tmp_psi = psi_gpu[state];
		psi_gpu[state] = psi_gpu[prev_state];
		psi_gpu[prev_state] = tmp_psi;
		if (minus_cnt & 1) psi_gpu[state] = make_cuDoubleComplex(-psi_gpu[state].x, -psi_gpu[state].y);
		if (i_cnt & 1){
			psi_gpu[prev_state] = make_cuDoubleComplex(psi_gpu[prev_state].y, psi_gpu[prev_state].x);
			psi_gpu[state] = make_cuDoubleComplex(psi_gpu[state].y, psi_gpu[state].x);
		}
		if ((i_cnt >> 1) & 1){
			psi_gpu[state] = make_cuDoubleComplex(-psi_gpu[state].x, -psi_gpu[state].y);
			psi_gpu[prev_state] = make_cuDoubleComplex(-psi_gpu[prev_state].x, -psi_gpu[prev_state].y);
		}
		minus_cnt = (num_pauli_op[2] - num_y1) + num_z1;
		if (minus_cnt & 1) psi_gpu[prev_state] = make_cuDoubleComplex(-psi_gpu[prev_state].x, -psi_gpu[prev_state].y);
	}
}


__host__ cudaError multi_Pauli_gate_host(int* gates, GTYPE *psi_gpu, ITYPE DIM, int n_qubits){
	cudaError cudaStatus;
	int num_pauli_op[4] = { 0, 0, 0, 0 };
	for (int i = 0; i < n_qubits; ++i) ++num_pauli_op[gates[i]];
	ITYPE bit_mask_Z = 0;
	for (int i = 0; i < n_qubits; ++i){
		if (gates[i] == 3) bit_mask_Z ^= (1 << i);
	}

	if (num_pauli_op[1] == 0 && num_pauli_op[2]==0){
		unsigned int block = DIM <= 1024 ? DIM : 1024;
		unsigned int grid = DIM / block;
		multi_Z_gate_gpu << <grid, block>> >(bit_mask_Z, DIM, psi_gpu);
		cudaStatus = cudaGetLastError();
		//cudaStatus = call_multi_Z_gate_gpu(gates, psi_gpu, DIM, n_qubits);
		return cudaStatus;
	}
	ITYPE bit_mask_XY = 0;
	for (int i = 0; i < n_qubits; ++i){
		if (gates[i] == 1 || gates[i]==2) bit_mask_XY ^= (1 << i);
	}

	int *gates_gpu, *num_pauli_op_gpu;
	checkCudaErrors(cudaMalloc((void**)&gates_gpu, 4 * sizeof(int)));
	checkCudaErrors(cudaMemcpy(gates_gpu, gates, 4 * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&num_pauli_op_gpu, 4 * sizeof(int)));
	checkCudaErrors(cudaMemcpy(num_pauli_op_gpu, num_pauli_op, 4 * sizeof(int), cudaMemcpyHostToDevice));
	
	ITYPE half_dim = DIM >> 1;
	unsigned int block = half_dim <= 1024 ? half_dim : 1024;
	unsigned int grid = half_dim / block;
	multi_Pauli_gate_gpu << <grid, block >> >(gates_gpu, bit_mask_XY, num_pauli_op_gpu, DIM, psi_gpu, n_qubits);
	cudaStatus = cudaGetLastError();

	cudaFree(gates_gpu);
	cudaFree(num_pauli_op_gpu);
	return cudaStatus;
}

