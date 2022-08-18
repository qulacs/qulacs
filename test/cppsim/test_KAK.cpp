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

TEST(KAKTest, random2bit) {
    QuantumGateBase* random_gate = gate::RandomUnitary({0, 1});
    auto KAK_ret = KAK_decomposition(random_gate);
}
