
#pragma once
/*
This KAK decomposition implementation is based on cirq implementation.
https://quantumai.google/reference/python/cirq/kak_decomposition
*/
#include <Eigen/Eigen>

#include "gate.hpp"
#include "gate_factory.hpp"
#include "gate_merge.hpp"
#include "general_quantum_operator.hpp"
#include "observable.hpp"
#include "state.hpp"

class KAK_data {
public:
    QuantumGateMatrix* single_qubit_operations_before[2];
    double interaction_coefficients[3];
    QuantumGateMatrix* single_qubit_operations_after[2];
};

// clang-format off
const Eigen::Matrix4cd KAK_MAGIC = (Eigen::Matrix4cd() <<
                                1,  0,  0, 1i,
                                0, 1i,  1,  0,
                                0, 1i, -1,  0,
                                1,  0,  0,-1i)
                            .finished() *sqrt(0.5);


const Eigen::Matrix4cd KAK_MAGIC_DAG = (Eigen::Matrix4cd() <<
                                1,  0,  0,  1,
                                0,-1i,-1i,  0,
                                0,  1, -1,  0,
                                -1i,0,  0, 1i)
                            .finished() *sqrt(0.5);

const Eigen::Matrix4cd KAK_GAMMA = (Eigen::Matrix4cd() <<
                                1,  1,  1,  1,
                                1,  1, -1, -1,
                               -1,  1, -1,  1,
                                1, -1, -1,  1)
                            .finished() *0.25;
// clang-format on

std::pair<Eigen::Matrix<CPPCTYPE, 2, 2>, Eigen::Matrix<CPPCTYPE, 2, 2>>
so4_to_magic_su2s(Eigen::Matrix4cd mat);

std::tuple<Eigen::Matrix4cd, Eigen::Matrix4cd>
bidiagonalize_real_matrix_pair_with_symmetric_products(
    Eigen::Matrix4d matA, Eigen::Matrix4d matB);

std::tuple<Eigen::Matrix4cd, Eigen::Matrix4cd, Eigen::Matrix4cd>
bidiagonalize_unitary_with_special_orthogonals(Eigen::Matrix4cd mat);

KAK_data KAK_decomposition_internal(QuantumGateBase* target_gate);

KAK_data KAK_decomposition(
    QuantumGateBase* target_gate, std::vector<UINT> target_bits);

void CSD_internal(ComplexMatrix mat, std::vector<UINT> now_control_qubits,
    std::vector<UINT> all_control_qubits, UINT ban,
    std::vector<QuantumGateBase*>& CSD_gate_list);

std::vector<QuantumGateBase*> CSD(QuantumGateBase* target_gate);
