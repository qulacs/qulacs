
#pragma once

#include <Eigen/Eigen>

#include "gate.hpp"
#include "gate_factory.hpp"
#include "general_quantum_operator.hpp"
#include "observable.hpp"
#include "state.hpp"

class KAK_data {
public:
    CPPCTYPE global_phase;
    QuantumGateMatrix* single_qubit_operations_before[2];
    CPPCTYPE interaction_coefficients[3];
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
                                1,  0,  0,-1i,
                                0,-1i,  1,  0,
                                0,-1i, -1,  0,
                                1,  0,  0, 1i)
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
    // ab[max_r][max_c] が　絶対値最大

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
    fb /= global;

    for (int gr = 0; gr < 4; gr++) {
        for (int gc = 0; gc < 4; gc++) {
            if (abs(ab(gr, gc) - fa(gr & 1, gc & 1) * fb(gr >> 1, gc >> 1)) >
                1e-6) {
                std::cerr << "bug" << __LINE__ << std::endl;
            }
        }
    }
    return make_pair(fa, fb);
}
std::tuple<Eigen::Matrix4cd, Eigen::Vector4d, Eigen::Matrix4cd>
bidiagonalize_unitary_with_special_orthogonals(Eigen::Matrix4cd mat) {
    mat(0, 1) += 1.0;
    std::cout << "mat=" << mat << std::endl;
    Eigen::JacobiSVD<Eigen::Matrix4cd,
        Eigen::ComputeThinU | Eigen::ComputeThinV>
        svd(mat);

    svd.computeU();
    svd.computeU();

    auto left = svd.matrixU();
    auto d = svd.singularValues();
    auto right = svd.matrixV();

    std::cout << "left=" << svd.matrixU() << std::endl;
    std::cout << "d=" << d << std::endl;
    std::cout << "right=" << right << std::endl;

    return std::make_tuple(left, d, right);
}

KAK_data KAK_decomposition(QuantumGateBase* target_gate) {
    // 入力は4*4 のゲート

    Eigen::Matrix4cd left, right;
    Eigen::Vector4d d;

    ComplexMatrix mat_moto;
    target_gate->set_matrix(mat_moto);
    Eigen::Matrix4cd mat = mat_moto;

    std::tie(left, d, right) = bidiagonalize_unitary_with_special_orthogonals(
        KAK_MAGIC_DAG * mat * KAK_MAGIC);

    Eigen::Matrix<CPPCTYPE, 2, 2> a1, a0, b1, b0;
    tie(a1, a0) = so4_to_magic_su2s(left);
    tie(b1, b0) = so4_to_magic_su2s(right);

    CPPCTYPE w, x, y, z;
    Eigen::Matrix<CPPCTYPE, 4, 1> d_diag_angle, wxyz;
    d_diag_angle[0] = std::arg(d[0]);
    d_diag_angle[1] = std::arg(d[1]);
    d_diag_angle[2] = std::arg(d[2]);
    d_diag_angle[3] = std::arg(d[3]);

    wxyz = KAK_GAMMA * d_diag_angle;

    KAK_data ans;
    ans.global_phase = std::exp(1.0i * wxyz[0]);
    ans.interaction_coefficients[0] = wxyz[1];
    ans.interaction_coefficients[1] = wxyz[2];
    ans.interaction_coefficients[2] = wxyz[3];
    QuantumGateMatrix* a0_gate =
        gate::DenseMatrix({target_gate->get_target_index_list()[0]}, a0);
    QuantumGateMatrix* a1_gate =
        gate::DenseMatrix({target_gate->get_target_index_list()[1]}, a1);
    QuantumGateMatrix* b0_gate =
        gate::DenseMatrix({target_gate->get_target_index_list()[0]}, b0);
    QuantumGateMatrix* b1_gate =
        gate::DenseMatrix({target_gate->get_target_index_list()[1]}, b1);

    ans.single_qubit_operations_after[0] = a0_gate;
    ans.single_qubit_operations_after[1] = a1_gate;
    ans.single_qubit_operations_before[0] = b0_gate;
    ans.single_qubit_operations_before[1] = b1_gate;

    return ans;
}