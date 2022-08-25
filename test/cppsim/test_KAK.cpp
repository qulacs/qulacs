#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <cppsim/KAK.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state_dm.hpp>
#include <cppsim/utility.hpp>
#include <csim/update_ops.hpp>
#include <fstream>
#include <functional>

#include "../util/util.hpp"

using namespace Eigen;
TEST(KAKTest, random2bit) {
    MatrixXf m = MatrixXf::Random(3, 2);
    std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
    JacobiSVD<MatrixXf> svd(m, ComputeThinU | ComputeThinV);
    std::cout << "Its singular values are:" << std::endl
              << svd.singularValues() << std::endl;
    std::cout
        << "Its left singular vectors are the columns of the thin U matrix:"
        << std::endl
        << svd.matrixU() << std::endl;
    std::cout
        << "Its right singular vectors are the columns of the thin V matrix:"
        << std::endl
        << svd.matrixV() << std::endl;
    Vector3f rhs(1, 0, 0);
    std::cout << "Now consider this rhs vector:" << std::endl
              << rhs << std::endl;
    std::cout << "A least-squares solution of m*x = rhs is:" << std::endl
              << svd.solve(rhs) << std::endl;

    QuantumGateBase* random_gate = gate::RandomUnitary({0, 1});
    auto KAK_ret = KAK_decomposition(random_gate);
}
