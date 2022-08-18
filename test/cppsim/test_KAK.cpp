#include <gtest/gtest.h>

#include <cppsim/KAK.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_named_pauli.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/type.hpp>
#include <cppsim/utility.hpp>
#include <csim/constant.hpp>
#include <fstream>

#include "../util/util.hpp"

using namespace std;
using namespace Eigen;
TEST(KAKTest, random2bit) {
    MatrixXf m = MatrixXf::Random(3, 2);
    cout << "Here is the matrix m:" << endl << m << endl;
    JacobiSVD<MatrixXf, ComputeThinU | ComputeThinV> svd(m);
    cout << "Its singular values are:" << endl << svd.singularValues() << endl;
    cout << "Its left singular vectors are the columns of the thin U matrix:"
         << endl
         << svd.matrixU() << endl;
    cout << "Its right singular vectors are the columns of the thin V matrix:"
         << endl
         << svd.matrixV() << endl;
    Vector3f rhs(1, 0, 0);
    cout << "Now consider this rhs vector:" << endl << rhs << endl;
    cout << "A least-squares solution of m*x = rhs is:" << endl
         << svd.solve(rhs) << endl;

    QuantumGateBase* random_gate = gate::RandomUnitary({0, 1});
    auto KAK_ret = KAK_decomposition(random_gate);
}
