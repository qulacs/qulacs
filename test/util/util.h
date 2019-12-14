#pragma once

#include <csim/type.h>
#include <Eigen/Core>
#include <string>

#ifdef __GNUC__
#if __GNUC__ >= 8
using namespace std::complex_literals;
#endif
#endif

// random
static UINT rand_int(UINT max) {
    return rand() % max;
}

static double rand_real() {
    return (rand() % RAND_MAX) / ((double)RAND_MAX);
}


// obtain single dense matrix
static Eigen::MatrixXcd get_eigen_matrix_single_Pauli(UINT pauli_id) {
    Eigen::MatrixXcd mat(2, 2);
    if (pauli_id == 0) mat << 1, 0, 0, 1;
    else if (pauli_id == 1) mat << 0, 1, 1, 0;
    else if (pauli_id == 2) mat << 0, -1.i, 1.i, 0;
    else if (pauli_id == 3) mat << 1, 0, 0, -1;
    return mat;
}

static Eigen::MatrixXcd get_eigen_matrix_random_single_qubit_unitary() {
    Eigen::MatrixXcd Identity, X, Y, Z;
    Identity = get_eigen_matrix_single_Pauli(0);
    X = get_eigen_matrix_single_Pauli(1);
    Y = get_eigen_matrix_single_Pauli(2);
    Z = get_eigen_matrix_single_Pauli(3);

    double icoef, xcoef, ycoef, zcoef, norm;
    icoef = rand_real(); xcoef = rand_real(); ycoef = rand_real(); zcoef = rand_real();
    norm = sqrt(icoef * icoef + xcoef + xcoef + ycoef * ycoef + zcoef * zcoef);
    icoef /= norm; xcoef /= norm; ycoef /= norm; zcoef /= norm;
    return icoef * Identity + 1.i*xcoef * X + 1.i*ycoef * Y + 1.i*zcoef * Z;
}


// expand matrix
static Eigen::MatrixXcd kronecker_product(const Eigen::MatrixXcd& lhs, const Eigen::MatrixXcd& rhs) {
    Eigen::MatrixXcd result(lhs.rows()*rhs.rows(), lhs.cols()*rhs.cols());
    for (int i = 0; i < lhs.cols(); i++) {
        for (int j = 0; j < lhs.rows(); j++) {
            result.block(i*rhs.rows(), j*rhs.cols(), rhs.rows(), rhs.cols()) = lhs(i, j)*rhs;
        }
    }
    return result;
}

static Eigen::MatrixXcd get_expanded_eigen_matrix_with_identity(UINT target_qubit_index, const Eigen::MatrixXcd& one_qubit_matrix, UINT qubit_count) {
    const ITYPE left_dim = 1ULL << target_qubit_index;
    const ITYPE right_dim = 1ULL << (qubit_count - target_qubit_index - 1);
    auto left_identity = Eigen::MatrixXcd::Identity(left_dim, left_dim);
    auto right_identity = Eigen::MatrixXcd::Identity(right_dim, right_dim);
    return kronecker_product(kronecker_product(right_identity, one_qubit_matrix), left_identity);
}

// get expanded matrix

static Eigen::MatrixXcd get_eigen_matrix_full_qubit_pauli(std::vector<UINT> pauli_ids) {
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Identity(1, 1);
    for (UINT i = 0; i < pauli_ids.size(); ++i) {
        result = kronecker_product(get_eigen_matrix_single_Pauli(pauli_ids[i]), result).eval();
    }
    return result;
}

static Eigen::MatrixXcd get_eigen_matrix_full_qubit_pauli(std::vector<UINT> index_list, std::vector<UINT> pauli_list, UINT qubit_count) {
    std::vector<UINT> whole_pauli_ids(qubit_count, 0);
    for (UINT i = 0; i < index_list.size(); ++i) {
        whole_pauli_ids[index_list[i]] = pauli_list[i];
    }
    return get_eigen_matrix_full_qubit_pauli(whole_pauli_ids);
}



static Eigen::MatrixXcd get_eigen_matrix_full_qubit_CNOT(UINT control_qubit_index, UINT target_qubit_index, UINT qubit_count) {
    ITYPE dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (ITYPE ind = 0; ind < dim; ++ind) {
        if (ind & (1ULL << control_qubit_index)) {
            result(ind, ind ^ (1ULL << target_qubit_index)) = 1;
        }
        else {
            result(ind, ind) = 1;
        }
    }
    return result;
}

static Eigen::MatrixXcd get_eigen_matrix_full_qubit_CZ(UINT control_qubit_index, UINT target_qubit_index, UINT qubit_count) {
    ITYPE dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (ITYPE ind = 0; ind < dim; ++ind) {
        if ((ind & (1ULL << control_qubit_index)) != 0 && (ind & (1ULL << target_qubit_index)) != 0) {
            result(ind, ind) = -1;
        }
        else {
            result(ind, ind) = 1;
        }
    }
    return result;
}

static Eigen::MatrixXcd get_eigen_matrix_full_qubit_SWAP(UINT target_qubit_index1, UINT target_qubit_index2, UINT qubit_count) {
    ITYPE dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (ITYPE ind = 0; ind < dim; ++ind) {
        bool flag1, flag2;
        flag1 = (ind&(1ULL << target_qubit_index1)) != 0;
        flag2 = (ind&(1ULL << target_qubit_index2)) != 0;
        if (flag1^flag2) {
            result(ind, ind ^ (1ULL << target_qubit_index1) ^ (1ULL << target_qubit_index2)) = 1;
        }
        else {
            result(ind, ind) = 1;
        }
    }
    return result;
}

// utils
static std::string convert_CTYPE_array_to_string(const CTYPE* state, ITYPE dim) {
    std::string str = "";
    for (ITYPE ind = 0; ind < dim; ++ind) {
        str += "(" + std::to_string(creal(state[ind])) + "," + std::to_string(cimag(state[ind])) + ") ";
    }
    return str;
}

static Eigen::VectorXcd convert_CTYPE_array_to_eigen_vector(const CTYPE* state, ITYPE dim) {
    Eigen::VectorXcd vec(dim);
    for (ITYPE i = 0; i < dim; ++i) vec[i] = state[i];
    return vec;
}

static void state_equal(const CTYPE* state, const Eigen::VectorXcd& test_state, ITYPE dim, std::string gate_string) {
    const double eps = 1e-14;
    Eigen::VectorXcd vec = convert_CTYPE_array_to_eigen_vector(state, dim);
    for (ITYPE ind = 0; ind < dim; ++ind) {
        ASSERT_NEAR(abs(vec[ind] - test_state[ind]), 0, eps) << gate_string << " at " << ind << std::endl << "Eigen : " << test_state.transpose() << std::endl << "CSIM : " << convert_CTYPE_array_to_string(state, dim) << std::endl;
    }
}
