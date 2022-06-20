#include <gtest/gtest.h>
#include "../util/util.hpp"

#include <cppsim/circuit.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state.hpp>

TEST(PauliOperatorTest, BasicTest) {
    int n = 10;
    double coef = 2.0;
    std::string Pauli_string = "X 0 Y 1 Z 2 I 3 X 4 Y 6 Z 5 I 8 X 7 Y 9";
    PauliOperator pauli = PauliOperator(Pauli_string, coef);
    QuantumState state = QuantumState(n);
    state.set_Haar_random_state();
    CPPCTYPE value = pauli.get_expectation_value(&state);
    ASSERT_NE(value, CPPCTYPE(0, 0));
}

TEST(PauliOperatorTest, EmptyString) {
    int n = 10;
    double coef = 2.0;
    std::string Pauli_string = "";
    PauliOperator pauli = PauliOperator(Pauli_string, coef);
    ASSERT_EQ(pauli.get_index_list().size(), 0);
}

TEST(PauliOperatorTest, PauliQubitOverflow) {
    int n = 2;
    double coef = 2.0;
    std::string Pauli_string = "X 0 X 1 X 3";
    PauliOperator pauli = PauliOperator(Pauli_string, coef);
    QuantumState state = QuantumState(n);
    state.set_Haar_random_state();
    EXPECT_THROW(
        pauli.get_expectation_value(&state),
        std::invalid_argument);
}

TEST(PauliOperatorTest, BrokenPauliString) {
    int n = 5;
    double coef = 2.0;
    std::string Pauli_string = "X 0 X Z 1 Y 2";
    EXPECT_THROW(PauliOperator(Pauli_string, coef), std::invalid_argument);
}

TEST(PauliOperatorTest, SpacedPauliString) {
    int n = 5;
    double coef = 2.0;
    std::string Pauli_string = "X 0 Y 1 ";
    PauliOperator pauli = PauliOperator(Pauli_string, coef);
    size_t PauliSize = pauli.get_index_list().size();
    ASSERT_EQ(PauliSize, 2);
}

TEST(PauliOperatorTest, PartedPauliString) {
    int n = 5;
    double coef = 2.0;
    std::string Pauli_string = "X 0 Y ";
    EXPECT_THROW(PauliOperator(Pauli_string, coef), std::invalid_argument);
}
