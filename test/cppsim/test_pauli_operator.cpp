#include <gtest/gtest.h>

#include <cppsim/pauli_operator.hpp>

struct PauliTestParam {
    PauliOperator op1;
    PauliOperator op2;
    PauliOperator expected;

    PauliTestParam(const PauliOperator op1, const PauliOperator op2,
        const PauliOperator expected)
        : op1(op1), op2(op2), expected(expected) {}
};

class PauliOperatorMultiplyTest
    : public testing::TestWithParam<PauliTestParam> {};

TEST_P(PauliOperatorMultiplyTest, MuliplyTest) {
    const auto p = GetParam();
    PauliOperator res = p.op1 * p.op2;
    EXPECT_EQ(p.expected.get_pauli_string(), res.get_pauli_string());
    EXPECT_EQ(p.expected.get_coef(), res.get_coef());
}

INSTANTIATE_TEST_CASE_P(SinglePauli, PauliOperatorMultiplyTest,
    testing::Values(PauliTestParam(PauliOperator("X", 2.0),
                        PauliOperator("X", 2.0), PauliOperator("I", 4.0)),
        PauliTestParam(PauliOperator("X", 2.0), PauliOperator("Y", 2.0),
            PauliOperator("Z", 4.0i)),
        PauliTestParam(PauliOperator("X", 2.0), PauliOperator("Z", 2.0),
            PauliOperator("Y", -4.0i)),
        PauliTestParam(PauliOperator("Y", 2.0), PauliOperator("X", 2.0),
            PauliOperator("Z", -4.0i)),
        PauliTestParam(PauliOperator("Y", 2.0), PauliOperator("Y", 2.0),
            PauliOperator("I", 4.0)),
        PauliTestParam(PauliOperator("Y", 2.0), PauliOperator("Z", 2.0),
            PauliOperator("X", 4.0i)),
        PauliTestParam(PauliOperator("Z", 2.0), PauliOperator("X", 2.0),
            PauliOperator("Y", 4.0i)),
        PauliTestParam(PauliOperator("Z", 2.0), PauliOperator("Y", 2.0),
            PauliOperator("X", -4.0i)),
        PauliTestParam(PauliOperator("Z", 2.0), PauliOperator("Z", 2.0),
            PauliOperator("I", 4.0))));

