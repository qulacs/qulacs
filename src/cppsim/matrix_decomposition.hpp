
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
Eigen::Matrix4cd KAK_MAGIC = (Eigen::Matrix4cd() <<  
                                1,  0,  0, 1i,
                                0, 1i,  1,  0,
                                0, 1i, -1,  0,
                                1,  0,  0,-1i)
                            .finished() *sqrt(0.5);


Eigen::Matrix4cd KAK_MAGIC_DAG = (Eigen::Matrix4cd() <<
                                1,  0,  0,  1,
                                0,-1i,-1i,  0,
                                0,  1, -1,  0,
                                -1i,0,  0, 1i)
                            .finished() *sqrt(0.5);

Eigen::Matrix4cd KAK_GAMMA = (Eigen::Matrix4cd() <<
                                1,  1,  1,  1,
                                1,  1, -1, -1,
                               -1,  1, -1,  1,
                                1, -1, -1,  1)
                            .finished() *0.25;
// clang-format on

std::pair<Eigen::Matrix<CPPCTYPE, 2, 2>, Eigen::Matrix<CPPCTYPE, 2, 2>>
so4_to_magic_su2s(Eigen::Matrix4cd mat) {
    Eigen::Matrix4cd ab = KAK_MAGIC * mat * KAK_MAGIC_DAG;
    Eigen::Matrix<CPPCTYPE, 2, 2> fa, fb;
    int max_r = 0, max_c = 0;
    for (int gr = 0; gr < 4; gr++) {
        for (int gc = 0; gc < 4; gc++) {
            if (abs(ab(max_r, max_c)) < abs(ab(gr, gc))) {
                max_r = gr;
                max_c = gc;
            }
        }
    }
    // ab[max_r][max_c] is abs max

    fa((max_r & 1), (max_c & 1)) = ab(max_r, max_c);
    fa((max_r & 1) ^ 1, (max_c & 1)) = ab(max_r ^ 1, max_c);
    fa((max_r & 1), (max_c & 1) ^ 1) = ab(max_r, max_c ^ 1);
    fa((max_r & 1) ^ 1, (max_c & 1) ^ 1) = ab(max_r ^ 1, max_c ^ 1);
    fb((max_r >> 1), (max_c >> 1)) = ab(max_r, max_c);
    fb((max_r >> 1) ^ 1, (max_c >> 1)) = ab(max_r ^ 2, max_c);
    fb((max_r >> 1), (max_c >> 1) ^ 1) = ab(max_r, max_c ^ 2);
    fb((max_r >> 1) ^ 1, (max_c >> 1) ^ 1) = ab(max_r ^ 2, max_c ^ 2);

    fa /= sqrt(fa(0, 0) * fa(1, 1) - fa(0, 1) * fa(1, 0));

    CPPCTYPE global = ab(max_r, max_c) / (fa((max_r & 1), (max_c & 1)) *
                                             fb((max_r >> 1), (max_c >> 1)));
    fb *= global;
    return make_pair(fa, fb);
}

std::tuple<Eigen::Matrix4cd, Eigen::Matrix4cd>
bidiagonalize_real_matrix_pair_with_symmetric_products(
    Eigen::Matrix4d matA, Eigen::Matrix4d matB) {
    // Bidiagonalize real matrix pairs
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(
        matA, Eigen::ComputeFullU | Eigen::ComputeFullV);

    if (abs(svd.singularValues()(3)) < 1e-11) {
        throw std::runtime_error(
            "The above KAK decomposition partially omit the implementation, "
            "and the function somtimes outputs invalid values. To address this "
            "problem, we apply a random matrix to the input unitary, perform "
            "the KAK decomposition, and retrieve the correct answer by "
            "applying its inverse. We expect that this is due to the "
            "degeneration of matrix eigenvalues.");
    }

    return std::make_tuple(svd.matrixU().transpose(), svd.matrixV());
}

std::tuple<Eigen::Matrix4cd, Eigen::Matrix4cd, Eigen::Matrix4cd>
bidiagonalize_unitary_with_special_orthogonals(Eigen::Matrix4cd mat) {
    Eigen::Matrix4d matA, matB;

    matA = mat.real();
    matB = mat.imag();
    auto left_and_right =
        bidiagonalize_real_matrix_pair_with_symmetric_products(matA, matB);
    Eigen::Matrix4cd left = std::get<0>(left_and_right);
    Eigen::Matrix4cd right = std::get<1>(left_and_right);
    Eigen::Matrix4cd diag = left * mat * right;

    if (left.determinant().real() < 0) {
        for (int i = 0; i < 4; i++) {
            left(0, i) *= -1;
        }
        diag(0, 0) *= -1;
    }
    if (right.determinant().real() < 0) {
        for (int i = 0; i < 4; i++) {
            right(i, 0) *= -1;
        }
        diag(0, 0) *= -1;
    }
    return std::make_tuple(left, diag, right);
}

KAK_data KAK_decomposition_internal(QuantumGateBase* target_gate) {
    // The input is assumed to be a gate with 4x4 matrix
    if (target_gate->get_target_index_list().size() != 2) {
        throw InvalidQubitCountException("target_gate index count must 2.");
    }

    Eigen::Matrix4cd left, diag, right;

    ComplexMatrix mat_original;
    target_gate->set_matrix(mat_original);
    Eigen::Matrix4cd mat = mat_original;

    std::tie(left, diag, right) =
        bidiagonalize_unitary_with_special_orthogonals(
            KAK_MAGIC_DAG * mat * KAK_MAGIC);

    Eigen::Matrix<CPPCTYPE, 2, 2> a1, a0, b1, b0;
    tie(a0, a1) = so4_to_magic_su2s(left.transpose());
    tie(b0, b1) = so4_to_magic_su2s(right.transpose());

    CPPCTYPE w, x, y, z;
    Eigen::Matrix<CPPCTYPE, 4, 1> d_diag_angle, wxyz;
    d_diag_angle[0] = std::arg(diag(0, 0));
    d_diag_angle[1] = std::arg(diag(1, 1));
    d_diag_angle[2] = std::arg(diag(2, 2));
    d_diag_angle[3] = std::arg(diag(3, 3));

    wxyz = KAK_GAMMA * d_diag_angle;

    KAK_data ans;
    ans.interaction_coefficients[0] = wxyz[1].real() * 2;
    ans.interaction_coefficients[1] = wxyz[2].real() * 2;
    ans.interaction_coefficients[2] = wxyz[3].real() * 2;

    a0 *= std::exp(1.0i * wxyz[0]);
    QuantumGateMatrix* a0_gate =
        gate::DenseMatrix(target_gate->get_target_index_list()[0], a0);
    QuantumGateMatrix* a1_gate =
        gate::DenseMatrix(target_gate->get_target_index_list()[1], a1);
    QuantumGateMatrix* b0_gate =
        gate::DenseMatrix(target_gate->get_target_index_list()[0], b0);
    QuantumGateMatrix* b1_gate =
        gate::DenseMatrix(target_gate->get_target_index_list()[1], b1);

    ans.single_qubit_operations_after[0] = a0_gate;
    ans.single_qubit_operations_after[1] = a1_gate;
    ans.single_qubit_operations_before[0] = b0_gate;
    ans.single_qubit_operations_before[1] = b1_gate;

    return ans;
}

KAK_data KAK_decomposition(
    QuantumGateBase* target_gate, std::vector<UINT> target_bits) {
    if (target_bits.size() != 2) {
        throw InvalidQubitCountException("Target qubit count must be 2");
    }
    if (target_bits[0] > target_bits[1]) {
        throw InvalidQubitCountException(
            "KAK decomposition assumes that the provided qubit indices are "
            "sorted, but unsorted arguments are given.");
    }

    /*
    上のKAK分解は実装を省略しているため、そのままだと縮退?
    などにより、正しい値が出ない可能性がありました。
    それを解消するため、分解したい行列に対して、各ビットにランダムユニタリを掛けてからKAK分解し、　その結果にさっきかけたランダムユニタリの逆行列を掛けることで対処しています.

    Since the KAK decomposition above omits the implementation, there is a
    possibility that the correct value will not come out due to degeneration
    etc. if it is as it is. In order to solve this problem, we deal with the
    matrix we want to decompose by multiplying each bit by random unitary, KAK
    decomposition, and multiplying the result by the inverse matrix of random
    unitary that we just applied.

    */

    QuantumGateBase* rand_gate0 = gate::RandomUnitary({target_bits[0]});
    QuantumGateBase* rand_gate1 = gate::RandomUnitary({target_bits[1]});
    QuantumGateBase* rand_gate01 = gate::merge(rand_gate0, rand_gate1);
    QuantumGateBase* merged_gate = gate::merge(target_gate, rand_gate01);
    delete rand_gate01;
    auto ans = KAK_decomposition_internal(merged_gate);
    delete merged_gate;
    auto grgate0 = gate::get_adjoint_gate(rand_gate0);
    auto after_gate0 =
        gate::merge(ans.single_qubit_operations_after[0], grgate0);
    delete ans.single_qubit_operations_after[0];
    ans.single_qubit_operations_after[0] = after_gate0;
    delete grgate0;
    delete rand_gate0;

    auto grgate1 = gate::get_adjoint_gate(rand_gate1);
    auto after_gate1 =
        gate::merge(ans.single_qubit_operations_after[1], grgate1);
    delete ans.single_qubit_operations_after[1];
    ans.single_qubit_operations_after[1] = after_gate1;
    delete grgate1;
    delete rand_gate1;

    return ans;
}

void CSD_internal(ComplexMatrix mat, std::vector<UINT> now_control_qubits,
    std::vector<UINT> all_control_qubits, int ban,
    std::vector<QuantumGateBase*>& CSD_gate_list) {
    UINT siz = mat.cols(), hsiz = siz / 2;
    if (siz == 4) {
        auto KAK_gate = gate::DenseMatrix(
            {all_control_qubits[0], all_control_qubits[1]}, mat);
        auto Kdata = KAK_decomposition(
            KAK_gate, {all_control_qubits[0], all_control_qubits[1]});
        for (auto it : now_control_qubits) {
            Kdata.single_qubit_operations_before[0]->add_control_qubit(it, 1);
        }
        CSD_gate_list.push_back(Kdata.single_qubit_operations_before[0]);
        for (auto it : now_control_qubits) {
            Kdata.single_qubit_operations_before[1]->add_control_qubit(it, 1);
        }
        CSD_gate_list.push_back(Kdata.single_qubit_operations_before[1]);

        auto itiXX =
            gate::PauliRotation({all_control_qubits[0], all_control_qubits[1]},
                {1, 1}, Kdata.interaction_coefficients[0]);
        auto matXX = gate::to_matrix_gate(itiXX);
        for (auto it : now_control_qubits) {
            matXX->add_control_qubit(it, 1);
        }
        CSD_gate_list.push_back(matXX);

        auto itiYY =
            gate::PauliRotation({all_control_qubits[0], all_control_qubits[1]},
                {2, 2}, Kdata.interaction_coefficients[1]);
        auto matYY = gate::to_matrix_gate(itiYY);
        for (auto it : now_control_qubits) {
            matYY->add_control_qubit(it, 1);
        }
        CSD_gate_list.push_back(matYY);

        auto itiZZ =
            gate::PauliRotation({all_control_qubits[0], all_control_qubits[1]},
                {3, 3}, Kdata.interaction_coefficients[2]);
        auto matZZ = gate::to_matrix_gate(itiZZ);
        for (auto it : now_control_qubits) {
            matZZ->add_control_qubit(it, 1);
        }
        CSD_gate_list.push_back(matZZ);

        for (auto it : now_control_qubits) {
            Kdata.single_qubit_operations_after[0]->add_control_qubit(it, 1);
        }
        CSD_gate_list.push_back(Kdata.single_qubit_operations_after[0]);
        for (auto it : now_control_qubits) {
            Kdata.single_qubit_operations_after[1]->add_control_qubit(it, 1);
        }
        CSD_gate_list.push_back(Kdata.single_qubit_operations_after[1]);

        delete itiXX;
        delete itiYY;
        delete itiZZ;

        delete KAK_gate;
        return;
    }
    ComplexMatrix Q11 = mat.block(0, 0, hsiz, hsiz);
    Eigen::JacobiSVD<ComplexMatrix> svd(
        Q11, Eigen::ComputeFullU | Eigen::ComputeFullV);
    ComplexMatrix Q12 = mat.block(0, hsiz, hsiz, hsiz);
    ComplexMatrix Q21 = mat.block(hsiz, 0, hsiz, hsiz);
    ComplexMatrix Q22 = mat.block(hsiz, hsiz, hsiz, hsiz);

    Eigen::HouseholderQR<ComplexMatrix> QR1(Q21 * svd.matrixV());
    Eigen::HouseholderQR<ComplexMatrix> QR2(Q12.adjoint() * svd.matrixU());
    ComplexMatrix U1 = svd.matrixU();
    ComplexMatrix U2 = QR1.householderQ();
    ComplexMatrix V1 = svd.matrixV();
    ComplexMatrix V2 = QR2.householderQ();

    ComplexMatrix R12 = V2.adjoint() * Q12.adjoint() * svd.matrixU();
    ComplexMatrix R21 = U2.adjoint() * Q21 * svd.matrixV();
    for (UINT i = 0; i < hsiz; i++) {
        if ((R12(i, i) * R21(i, i)).real() > 0) {
            R12(i, i) *= -1;
            for (UINT j = 0; j < hsiz; j++) {
                V2(j, i) *= -1;
            }
        }
    }

    auto added_control_qubits = now_control_qubits;
    added_control_qubits.push_back(all_control_qubits[ban]);
    CSD_internal(V1.adjoint(), now_control_qubits, all_control_qubits, ban - 1,
        CSD_gate_list);
    CSD_internal(V2.adjoint() * V1, added_control_qubits, all_control_qubits,
        ban - 1, CSD_gate_list);

    std::vector<double> args(hsiz);
    for (UINT i = 0; i < hsiz; i++) {
        args[i] = asin(R12(i, i).real()) * 2;
    }
    // Fast Mobius transformation
    for (UINT h = 0; h < ban; h++) {
        for (UINT i = 0; i < hsiz; i++) {
            if (i & (1 << h)) {
                args[i] -= args[i - (1 << h)];
            }
        }
    }
    for (UINT i = 0; i < hsiz; i++) {
        auto CSD_control_qubits = now_control_qubits;
        for (UINT j = 0; j < ban; j++) {
            if (i & (1 << j)) {
                CSD_control_qubits.push_back(all_control_qubits[j]);
            }
        }
        auto itiRY = gate::RY(all_control_qubits[ban], args[i]);
        auto matRY = gate::to_matrix_gate(itiRY);
        for (auto it : CSD_control_qubits) {
            matRY->add_control_qubit(it, 1);
        }
        CSD_gate_list.push_back(matRY);
        delete itiRY;
    }
    CSD_internal(
        U1, now_control_qubits, all_control_qubits, ban - 1, CSD_gate_list);
    CSD_internal(U2 * U1.adjoint(), added_control_qubits, all_control_qubits,
        ban - 1, CSD_gate_list);
}

std::vector<QuantumGateBase*> CSD(QuantumGateBase* target_gate) {
    if (target_gate->get_control_index_list().size() > 0) {
        throw InvalidQubitCountException(
            "do not include control qubits in target_gate");
    }
    if (target_gate->get_target_index_list().size() < 2) {
        throw InvalidQubitCountException(
            "CSD qubit size must be no less than 2");
    }
    std::vector<QuantumGateBase*> ans;
    ComplexMatrix mat;
    target_gate->set_matrix(mat);
    CSD_internal(mat, {}, target_gate->get_target_index_list(),
        target_gate->get_target_index_list().size() - 1, ans);
    return ans;
}