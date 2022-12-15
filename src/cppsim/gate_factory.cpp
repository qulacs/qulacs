
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#define _USE_MATH_DEFINES
#include "gate_factory.hpp"

#include <Eigen/QR>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "exception.hpp"
#include "gate.hpp"
#include "gate_matrix.hpp"
#include "gate_matrix_diagonal.hpp"
#include "gate_matrix_sparse.hpp"
#include "gate_merge.hpp"
#include "gate_named_one.hpp"
#include "gate_named_pauli.hpp"
#include "gate_named_two.hpp"
#include "gate_reflect.hpp"
#include "gate_reversible.hpp"
#include "type.hpp"

namespace gate {
ComplexMatrix get_IBMQ_matrix(double theta, double phi, double lambda);

ClsOneQubitGate* Identity(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->IGateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* X(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->XGateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* Y(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->YGateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* Z(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->ZGateinit(qubit_index);
    return ptr;
}
ClsOneQubitGate* H(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->HGateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* S(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->SGateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* Sdag(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->SdagGateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* T(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->TGateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* Tdag(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->TdagGateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* sqrtX(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->sqrtXGateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* sqrtXdag(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->sqrtXdagGateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* sqrtY(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->sqrtYGateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* sqrtYdag(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->sqrtYdagGateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* P0(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->P0Gateinit(qubit_index);
    return ptr;
}

ClsOneQubitGate* P1(UINT qubit_index) {
    auto ptr = new ClsOneQubitGate();
    ptr->P1Gateinit(qubit_index);
    return ptr;
}

ClsOneQubitRotationGate* RX(UINT qubit_index, double angle) {
    auto ptr = new ClsOneQubitRotationGate();
    ptr->RXGateinit(qubit_index, angle);
    return ptr;
}
ClsOneQubitRotationGate* RY(UINT qubit_index, double angle) {
    auto ptr = new ClsOneQubitRotationGate();
    ptr->RYGateinit(qubit_index, angle);
    return ptr;
}
ClsOneQubitRotationGate* RZ(UINT qubit_index, double angle) {
    auto ptr = new ClsOneQubitRotationGate();
    ptr->RZGateinit(qubit_index, angle);
    return ptr;
}
ClsOneQubitRotationGate* RotInvX(UINT qubit_index, double angle) {
    auto ptr = new ClsOneQubitRotationGate();
    ptr->RXGateinit(qubit_index, angle);
    return ptr;
}
ClsOneQubitRotationGate* RotInvY(UINT qubit_index, double angle) {
    auto ptr = new ClsOneQubitRotationGate();
    ptr->RYGateinit(qubit_index, angle);
    return ptr;
}
ClsOneQubitRotationGate* RotInvZ(UINT qubit_index, double angle) {
    auto ptr = new ClsOneQubitRotationGate();
    ptr->RZGateinit(qubit_index, angle);
    return ptr;
}
ClsOneQubitRotationGate* RotX(UINT qubit_index, double angle) {
    auto ptr = new ClsOneQubitRotationGate();
    ptr->RXGateinit(qubit_index, -angle);
    return ptr;
}
ClsOneQubitRotationGate* RotY(UINT qubit_index, double angle) {
    auto ptr = new ClsOneQubitRotationGate();
    ptr->RYGateinit(qubit_index, -angle);
    return ptr;
}
ClsOneQubitRotationGate* RotZ(UINT qubit_index, double angle) {
    auto ptr = new ClsOneQubitRotationGate();
    ptr->RZGateinit(qubit_index, -angle);
    return ptr;
}

ComplexMatrix get_IBMQ_matrix(double theta, double phi, double lambda) {
    CPPCTYPE im(0, 1);
    CPPCTYPE exp_val1 = exp(im * phi);
    CPPCTYPE exp_val2 = exp(im * lambda);
    CPPCTYPE cos_val = cos(theta / 2);
    CPPCTYPE sin_val = sin(theta / 2);

    ComplexMatrix matrix(2, 2);
    matrix(0, 0) = cos_val;
    matrix(0, 1) = -exp_val2 * sin_val;
    matrix(1, 0) = exp_val1 * sin_val;
    matrix(1, 1) = exp_val1 * exp_val2 * cos_val;
    return matrix;
}
QuantumGateMatrix* U1(UINT qubit_index, double lambda) {
    ComplexMatrix matrix = get_IBMQ_matrix(0, 0, lambda);
    std::vector<UINT> vec;
    vec.push_back(qubit_index);
    return new QuantumGateMatrix(vec, matrix);
}
QuantumGateMatrix* U2(UINT qubit_index, double phi, double lambda) {
    ComplexMatrix matrix = get_IBMQ_matrix(M_PI / 2, phi, lambda);
    std::vector<UINT> vec;
    vec.push_back(qubit_index);
    return new QuantumGateMatrix(vec, matrix);
}
QuantumGateMatrix* U3(
    UINT qubit_index, double theta, double phi, double lambda) {
    ComplexMatrix matrix = get_IBMQ_matrix(theta, phi, lambda);
    std::vector<UINT> vec;
    vec.push_back(qubit_index);
    return new QuantumGateMatrix(vec, matrix);
}

ClsOneControlOneTargetGate* CNOT(
    UINT control_qubit_index, UINT target_qubit_index) {
    if (control_qubit_index == target_qubit_index) {
        throw InvalidControlQubitException(
            "Error: gate::CNOT(UINT, UINT): control_qubit_index and "
            "target_qubit_index has the same value."
            "\nInfo: NULL used to be returned, "
            "but it changed to throw exception.");
    }
    auto ptr = new ClsOneControlOneTargetGate();
    ptr->CNOTGateinit(control_qubit_index, target_qubit_index);
    return ptr;
}

ClsOneControlOneTargetGate* CZ(
    UINT control_qubit_index, UINT target_qubit_index) {
    if (control_qubit_index == target_qubit_index) {
        throw InvalidControlQubitException(
            "Error: gate::CZ(UINT, UINT): control_qubit_index and "
            "target_qubit_index has the same value."
            "\nInfo: NULL used to be returned, "
            "but it changed to throw exception.");
    }
    auto ptr = new ClsOneControlOneTargetGate();
    ptr->CZGateinit(control_qubit_index, target_qubit_index);
    return ptr;
}
ClsTwoQubitGate* SWAP(UINT qubit_index1, UINT qubit_index2) {
    if (qubit_index1 == qubit_index2) {
        throw DuplicatedQubitIndexException(
            "Error: gate::SWAP(UINT, UINT): two indices have the same value."
            "\nInfo: NULL used to be returned, "
            "but it changed to throw exception.");
    }
    auto ptr = new ClsTwoQubitGate();
    ptr->SWAPGateinit(qubit_index1, qubit_index2);
    return ptr;
}

ClsPauliGate* Pauli(std::vector<UINT> target, std::vector<UINT> pauli_id) {
    if (!check_is_unique_index_list(target)) {
        throw DuplicatedQubitIndexException(
            "Error: gate::Pauli(std::vector<UINT> target, "
            "std::vector<UINT>pauli_id): target list contains "
            "duplicated values."
            "\nInfo: NULL used to be returned, "
            "but it changed to throw exception.");
    }
    auto pauli = new PauliOperator(target, pauli_id);
    return new ClsPauliGate(pauli);
}
ClsPauliRotationGate* PauliRotation(
    std::vector<UINT> target, std::vector<UINT> pauli_id, double angle) {
    if (!check_is_unique_index_list(target)) {
        throw DuplicatedQubitIndexException(
            "Error: gate::PauliRotation(std::vector<UINT> target, "
            "std::vector<UINT>pauli_id, double angle): target list "
            "contains duplicated values."
            "\nInfo: NULL used to be returned, "
            "but it changed to throw exception.");
    }
    auto pauli = new PauliOperator(target, pauli_id, angle);
    return new ClsPauliRotationGate(angle, pauli);
}

QuantumGateMatrix* DenseMatrix(UINT target_index, ComplexMatrix matrix) {
    std::vector<UINT> target_list(1, target_index);
    return new QuantumGateMatrix(target_list, matrix);
}
QuantumGateMatrix* DenseMatrix(
    std::vector<UINT> target_list, ComplexMatrix matrix) {
    if (!check_is_unique_index_list(target_list)) {
        throw DuplicatedQubitIndexException(
            "Error: gate::DenseMatrix(std::vector<UINT> target_list, "
            "ComplexMatrix matrix): target list contains duplicated values."
            "\nInfo: NULL used to be returned, "
            "but it changed to throw exception.");
    }
    return new QuantumGateMatrix(target_list, matrix);
}

QuantumGateSparseMatrix* SparseMatrix(
    std::vector<UINT> target_list, SparseComplexMatrix matrix) {
    if (!check_is_unique_index_list(target_list)) {
        throw DuplicatedQubitIndexException(
            "Error: gate::SparseMatrix(std::vector<UINT> target_list, "
            "SparseComplexMatrix matrix): target list contains duplicated "
            "values."
            "\nInfo: NULL used to be returned, "
            "but it changed to throw exception.");
    }
    return new QuantumGateSparseMatrix(target_list, matrix);
}
QuantumGateDiagonalMatrix* DiagonalMatrix(
    std::vector<UINT> target_list, ComplexVector diagonal_element) {
    if (!check_is_unique_index_list(target_list)) {
        throw DuplicatedQubitIndexException(
            "Error: gate::DiagonalMatrix(std::vector<UINT> target_list, "
            "ComplexVector diagonal_element): target list contains "
            "duplicated values."
            "\nInfo: NULL used to be returned, "
            "but it changed to throw exception.");
    }
    return new QuantumGateDiagonalMatrix(target_list, diagonal_element);
}

QuantumGateMatrix* RandomUnitary(std::vector<UINT> target_list) {
    if (!check_is_unique_index_list(target_list)) {
        throw DuplicatedQubitIndexException(
            "Error: gate::RandomUnitary(std::vector<UINT> target_list): "
            "target list contains duplicated values."
            "\nInfo: NULL used to be returned, "
            "but it changed to throw exception.");
    }
    Random random;
    UINT qubit_count = (UINT)target_list.size();
    ITYPE dim = 1ULL << qubit_count;
    ComplexMatrix matrix(dim, dim);
    for (ITYPE i = 0; i < dim; ++i) {
        for (ITYPE j = 0; j < dim; ++j) {
            matrix(i, j) = (random.normal() + 1.i * random.normal()) / sqrt(2.);
        }
    }
    Eigen::HouseholderQR<ComplexMatrix> qr_solver(matrix);
    ComplexMatrix Q = qr_solver.householderQ();
    // actual R matrix is upper-right triangle of matrixQR
    auto R = qr_solver.matrixQR();
    for (ITYPE i = 0; i < dim; ++i) {
        CPPCTYPE phase = R(i, i) / abs(R(i, i));
        for (ITYPE j = 0; j < dim; ++j) {
            Q(j, i) *= phase;
        }
    }
    return new QuantumGateMatrix(target_list, Q);
}
QuantumGateMatrix* RandomUnitary(std::vector<UINT> target_list, UINT seed) {
    if (!check_is_unique_index_list(target_list)) {
        throw DuplicatedQubitIndexException(
            "Error: gate::RandomUnitary(std::vector<UINT> target_list): "
            "target list contains duplicated values."
            "\nInfo: NULL used to be returned, "
            "but it changed to throw exception.");
    }
    Random random;
    random.set_seed(seed);
    UINT qubit_count = (UINT)target_list.size();
    ITYPE dim = 1ULL << qubit_count;
    ComplexMatrix matrix(dim, dim);
    for (ITYPE i = 0; i < dim; ++i) {
        for (ITYPE j = 0; j < dim; ++j) {
            matrix(i, j) = (random.normal() + 1.i * random.normal()) / sqrt(2.);
        }
    }
    Eigen::HouseholderQR<ComplexMatrix> qr_solver(matrix);
    ComplexMatrix Q = qr_solver.householderQ();
    // actual R matrix is upper-right triangle of matrixQR
    auto R = qr_solver.matrixQR();
    for (ITYPE i = 0; i < dim; ++i) {
        CPPCTYPE phase = R(i, i) / abs(R(i, i));
        for (ITYPE j = 0; j < dim; ++j) {
            Q(j, i) *= phase;
        }
    }
    return new QuantumGateMatrix(target_list, Q);
}
ClsReversibleBooleanGate* ReversibleBoolean(
    std::vector<UINT> target_qubit_index_list,
    std::function<ITYPE(ITYPE, ITYPE)> function_ptr) {
    if (!check_is_unique_index_list(target_qubit_index_list)) {
        throw DuplicatedQubitIndexException(
            "Error: gate::ReversibleBoolean(std::vector<UINT> "
            "target_qubit_index_list, std::function<ITYPE(ITYPE,ITYPE)> "
            "function_ptr): target list contains duplicated values."
            "\nInfo: NULL used to be returned, "
            "but it changed to throw exception.");
    }
    return new ClsReversibleBooleanGate(target_qubit_index_list, function_ptr);
}
ClsStateReflectionGate* StateReflection(
    const QuantumStateBase* reflection_state) {
    return new ClsStateReflectionGate(reflection_state);
}

QuantumGate_Probabilistic* BitFlipNoise(UINT target_index, double prob) {
    auto gate0 = X(target_index);
    auto gate1 = Identity(target_index);
    auto new_gate =
        new QuantumGate_Probabilistic({prob, 1 - prob}, {gate0, gate1});
    delete gate0;
    delete gate1;
    return new_gate;
}
QuantumGate_Probabilistic* DephasingNoise(UINT target_index, double prob) {
    auto gate0 = Z(target_index);
    auto gate1 = Identity(target_index);
    auto new_gate =
        new QuantumGate_Probabilistic({prob, 1 - prob}, {gate0, gate1});
    delete gate0;
    delete gate1;
    return new_gate;
}
QuantumGate_Probabilistic* IndependentXZNoise(UINT target_index, double prob) {
    auto gate0 = X(target_index);
    auto gate1 = Z(target_index);
    auto gate2 = Y(target_index);
    auto gate3 = Identity(target_index);
    double p1 = prob * (1 - prob);
    double p2 = prob * prob;
    auto new_gate = new QuantumGate_Probabilistic(
        {p1, p1, p2, 1 - 2 * p1 - p2}, {gate0, gate1, gate2, gate3});
    delete gate0;
    delete gate1;
    delete gate2;
    delete gate3;
    return new_gate;
}
QuantumGate_Probabilistic* DepolarizingNoise(UINT target_index, double prob) {
    auto gate0 = X(target_index);
    auto gate1 = Z(target_index);
    auto gate2 = Y(target_index);
    auto gate3 = Identity(target_index);
    auto new_gate = new QuantumGate_Probabilistic(
        {prob / 3, prob / 3, prob / 3, 1 - prob}, {gate0, gate1, gate2, gate3});
    delete gate0;
    delete gate1;
    delete gate2;
    delete gate3;
    return new_gate;
}
QuantumGate_Probabilistic* TwoQubitDepolarizingNoise(
    UINT target_index1, UINT target_index2, double prob) {
    if (target_index1 == target_index2) {
        throw DuplicatedQubitIndexException(
            "Error: gate::TwoQubitDepolarizingNoise(UINT, UINT, double): "
            "target list contains duplicated values."
            "\nInfo: NULL used to be returned, "
            "but it changed to throw exception.");
    }
    std::vector<QuantumGateBase*> gate_list;
    for (int i = 0; i < 16; ++i) {
        if (i != 0) {
            UINT pauli_qubit1 = i % 4;
            UINT pauli_qubit2 = i / 4;
            auto gate_pauli = Pauli(
                {target_index1, target_index2}, {pauli_qubit1, pauli_qubit2});
            auto gate_dense = gate::to_matrix_gate(gate_pauli);
            gate_list.push_back(gate_dense);
            delete gate_pauli;
        } else {
            gate_list.push_back(Identity(target_index1));
        }
    }
    std::vector<double> probabilities(16, prob / 15);
    probabilities[0] = 1 - prob;
    auto new_gate = new QuantumGate_Probabilistic(probabilities, gate_list);
    for (UINT gate_index = 0; gate_index < 16; ++gate_index) {
        delete gate_list[gate_index];
    }
    return new_gate;
}
QuantumGate_CPTP* AmplitudeDampingNoise(UINT target_index, double prob) {
    ComplexMatrix damping_matrix_0(2, 2), damping_matrix_1(2, 2);
    damping_matrix_0 << 1, 0, 0, sqrt(1 - prob);
    damping_matrix_1 << 0, sqrt(prob), 0, 0;
    auto gate0 = DenseMatrix({target_index}, damping_matrix_0);
    auto gate1 = DenseMatrix({target_index}, damping_matrix_1);
    auto new_gate = new QuantumGate_CPTP({gate0, gate1});
    delete gate0;
    delete gate1;
    return new_gate;
}
QuantumGate_Instrument* Measurement(
    UINT target_index, UINT classical_register_address) {
    auto gate0 = P0(target_index);
    auto gate1 = P1(target_index);
    auto new_gate =
        new QuantumGate_Instrument({gate0, gate1}, classical_register_address);
    delete gate0;
    delete gate1;
    return new_gate;
}

ClsNoisyEvolution* NoisyEvolution(Observable* hamiltonian,
    std::vector<GeneralQuantumOperator*> c_ops, double time, double dt) {
    return new ClsNoisyEvolution(hamiltonian, c_ops, time, dt);
}
ClsNoisyEvolution_fast* NoisyEvolution_fast(Observable* hamiltonian,
    std::vector<GeneralQuantumOperator*> c_ops, double time) {
    return new ClsNoisyEvolution_fast(hamiltonian, c_ops, time);
}

ClsNoisyEvolution_auto* NoisyEvolution_auto(Observable* hamiltonian,
    std::vector<GeneralQuantumOperator*> c_ops, double time) {
    return new ClsNoisyEvolution_auto(hamiltonian, c_ops, time);
}
QuantumGateBase* create_quantum_gate_from_string(std::string gate_string) {
    const char* gateString = gate_string.c_str();
    char* sbuf;
    const char delim[] = " ";
    std::vector<UINT> targets;
    QuantumGateBase* gate = NULL;
    char* buf = (char*)calloc(strlen(gateString) + 1, sizeof(char));
    strcpy(buf, gateString);
    sbuf = strtok(buf, delim);

    if (strcasecmp(sbuf, "I") == 0)
        gate = gate::Identity(atoi(strtok(NULL, delim)));
    else if (strcasecmp(sbuf, "X") == 0)
        gate = gate::X(atoi(strtok(NULL, delim)));
    else if (strcasecmp(sbuf, "Y") == 0)
        gate = gate::Y(atoi(strtok(NULL, delim)));
    else if (strcasecmp(sbuf, "Z") == 0)
        gate = gate::Z(atoi(strtok(NULL, delim)));
    else if (strcasecmp(sbuf, "H") == 0)
        gate = gate::H(atoi(strtok(NULL, delim)));
    else if (strcasecmp(sbuf, "S") == 0)
        gate = gate::S(atoi(strtok(NULL, delim)));
    else if (strcasecmp(sbuf, "Sdag") == 0)
        gate = gate::Sdag(atoi(strtok(NULL, delim)));
    else if (strcasecmp(sbuf, "T") == 0)
        gate = gate::T(atoi(strtok(NULL, delim)));
    else if (strcasecmp(sbuf, "Tdag") == 0)
        gate = gate::Tdag(atoi(strtok(NULL, delim)));
    else if (strcasecmp(sbuf, "CNOT") == 0 || strcasecmp(sbuf, "CX") == 0) {
        unsigned int control = atoi(strtok(NULL, delim));
        unsigned int target = atoi(strtok(NULL, delim));
        gate = gate::CNOT(control, target);
    } else if (strcasecmp(sbuf, "CZ") == 0) {
        unsigned int control = atoi(strtok(NULL, delim));
        unsigned int target = atoi(strtok(NULL, delim));
        gate = gate::CZ(control, target);
    } else if (strcasecmp(sbuf, "SWAP") == 0) {
        unsigned int target1 = atoi(strtok(NULL, delim));
        unsigned int target2 = atoi(strtok(NULL, delim));
        gate = gate::SWAP(target1, target2);
    } else if (strcasecmp(sbuf, "U1") == 0) {
        unsigned int target = atoi(strtok(NULL, delim));
        double theta1 = atof(strtok(NULL, delim));
        gate = gate::U1(target, theta1);
    } else if (strcasecmp(sbuf, "U2") == 0) {
        unsigned int target = atoi(strtok(NULL, delim));
        double theta1 = atof(strtok(NULL, delim));
        double theta2 = atof(strtok(NULL, delim));
        gate = gate::U2(target, theta1, theta2);
    } else if (strcasecmp(sbuf, "U3") == 0) {
        unsigned int target = atoi(strtok(NULL, delim));
        double theta1 = atof(strtok(NULL, delim));
        double theta2 = atof(strtok(NULL, delim));
        double theta3 = atof(strtok(NULL, delim));
        gate = gate::U3(target, theta1, theta2, theta3);
    } else if (strcasecmp(sbuf, "RX") == 0) {
        unsigned int target = atoi(strtok(NULL, delim));
        double theta = atof(strtok(NULL, delim));
        gate = gate::RX(target, theta);
    } else if (strcasecmp(sbuf, "RY") == 0) {
        unsigned int target = atoi(strtok(NULL, delim));
        double theta = atof(strtok(NULL, delim));
        gate = gate::RY(target, theta);
    } else if (strcasecmp(sbuf, "RZ") == 0) {
        unsigned int target = atoi(strtok(NULL, delim));
        double theta = atof(strtok(NULL, delim));
        gate = gate::RZ(target, theta);
    } else if (strcasecmp(sbuf, "RM") == 0) {
        char* pauliStr = strtok(NULL, delim);
        unsigned int targetCount = (UINT)strlen(pauliStr);

        std::vector<UINT> pauli(targetCount, 0);
        for (unsigned int i = 0; i < targetCount; i++) {
            if (pauliStr[i] == 'x' || pauliStr[i] == 'X')
                pauli[i] = 1;
            else if (pauliStr[i] == 'y' || pauliStr[i] == 'Y')
                pauli[i] = 2;
            else if (pauliStr[i] == 'z' || pauliStr[i] == 'Z')
                pauli[i] = 3;
        }

        targets = std::vector<UINT>(targetCount, 0);
        for (unsigned int i = 0; i < targetCount; i++) {
            targets[i] = atoi(strtok(NULL, delim));
        }

        double theta = atof(strtok(NULL, delim));
        gate = gate::PauliRotation(targets, pauli, theta);
    } else if (strcasecmp(sbuf, "U") == 0) {
        unsigned int targetCount = atoi(strtok(NULL, delim));

        targets = std::vector<UINT>(targetCount, 0);
        for (unsigned int i = 0; i < targetCount; i++) {
            targets[i] = atoi(strtok(NULL, delim));
        }
        ITYPE dim = 1ULL << targetCount;
        ComplexMatrix matrix(dim, dim);

        for (ITYPE i = 0; i < dim * dim; i++) {
            char* token;
            token = strtok(NULL, delim);
            matrix(i / dim, i % dim) = atof(token);
            token = strtok(NULL, delim);
            matrix(i / dim, i % dim) += CPPCTYPE(0, 1) * atof(token);
        }
        gate = gate::DenseMatrix(targets, matrix);
    }
    free(buf);
    return gate;
}

}  // namespace gate
