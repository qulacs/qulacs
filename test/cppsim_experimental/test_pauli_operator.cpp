#include <gtest/gtest.h>

#include <cppsim_experimental/pauli_operator.hpp>

TEST(PauliOperatorTest, OperatorEQTest) {
    MultiQubitPauliOperator op1("X 0 X 1 X 2");
    MultiQubitPauliOperator op2("X 0 X 1 X 2");
    MultiQubitPauliOperator op3("Y 0 X 1 X 2");
    MultiQubitPauliOperator op4("X 0 X 1");

    EXPECT_TRUE(op1 == op2);
    EXPECT_FALSE(op1 == op3);
    EXPECT_FALSE(op1 == op4);
}