#include <algorithm>
#include <cassert>
#include <functional>
#include <sstream>

#include "gate.hpp"
#include "general_quantum_operator.hpp"

GeneralQuantumOperator* to_general_quantum_operator(
    const QuantumGateBase* gate, UINT GQO_qubits, double tol = 1e-6) {
    // 返り値のqubitの数を指定
    if (gate->get_control_index_list().size() > 0) {
        throw std::runtime_error("gate must not have control qubit. ");
    }
    ComplexMatrix mat;
    gate->set_matrix(mat);
    UINT n = gate->get_target_index_list().size();
    for (UINT h = 0; h < n; h++) {
        UINT b = (1 << h);
        for (UINT i = 0; i < (1 << n); i++) {
            if (i & b) {
                continue;
            }
            for (UINT j = 0; j < (1 << n); j++) {
                if (j & b) {
                    continue;
                }
                CPPCTYPE UL_I = mat(i, j) + mat(i + b, j + b);
                CPPCTYPE UR_X = mat(i, j + b) + mat(i + b, j);
                CPPCTYPE DL_Y = (mat(i, j + b) - mat(i + b, j)) * 1.0i;
                CPPCTYPE DR_Z = mat(i, j) - mat(i + b, j + b);
                mat(i, j) = UL_I;
                mat(i, j + b) = UR_X;
                mat(i + b, j) = DL_Y;
                mat(i + b, j + b) = DR_Z;
            }
        }
    }
    auto ans = new GeneralQuantumOperator(GQO_qubits);
    double waru = (1 << n);
    for (UINT i = 0; i < (1 << n); i++) {
        for (UINT j = 0; j < (1 << n); j++) {
            if (abs(mat(i, j)) <= tol) {
                continue;
            }
            std::vector<UINT> index_list;
            std::vector<UINT> pauli_list;
            for (UINT h = 0; h < n; h++) {
                if (i & (1 << h)) {
                    if (j & (1 << h)) {  // Z
                        index_list.push_back(gate->get_target_index_list()[h]);
                        pauli_list.push_back(3);
                    } else {  // Y
                        index_list.push_back(gate->get_target_index_list()[h]);
                        pauli_list.push_back(2);
                    }
                } else if (j & (1 << h)) {  // X
                    index_list.push_back(gate->get_target_index_list()[h]);
                    pauli_list.push_back(1);
                }
            }
            ans->add_operator(index_list, pauli_list, mat(i, j) / waru);
        }
    }

    return ans;
}
