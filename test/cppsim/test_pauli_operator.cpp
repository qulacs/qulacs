#include <gtest/gtest.h>

#include <cppsim/pauli_operator.hpp>

struct PauliTestParam {
    std::string test_name;
    PauliOperator op1;
    PauliOperator op2;
    PauliOperator expected;

    PauliTestParam(const std::string& test_name, const PauliOperator op1,
        const PauliOperator op2, const PauliOperator expected)
        : test_name(test_name), op1(op1), op2(op2), expected(expected) {}
};

std::ostream& operator<<(std::ostream& stream, const PauliTestParam& p) {
    return stream << p.test_name;
}

class PauliOperatorMultiplyTest
    : public testing::TestWithParam<PauliTestParam> {};

TEST_P(PauliOperatorMultiplyTest, MultiplyTest) {
    const auto p = GetParam();
    PauliOperator res = p.op1 * p.op2;
    EXPECT_EQ(p.expected.get_pauli_string(), res.get_pauli_string());
    EXPECT_EQ(p.expected.get_coef(), res.get_coef());
}

TEST_P(PauliOperatorMultiplyTest, MultiplyAssignmentTest) {
    const auto p = GetParam();
    PauliOperator res = p.op1;
    res *= p.op2;
    EXPECT_EQ(p.expected.get_pauli_string(), res.get_pauli_string());
    EXPECT_EQ(p.expected.get_coef(), res.get_coef());
}

INSTANTIATE_TEST_CASE_P(SinglePauli, PauliOperatorMultiplyTest,
    testing::Values(PauliTestParam("XX", PauliOperator("X 0", 2.0),
                        PauliOperator("X 0", 2.0), PauliOperator("I 0", 4.0)),
        PauliTestParam("XY", PauliOperator("X 0", 2.0),
            PauliOperator("Y 0", 2.0), PauliOperator("Z 0", 4.0i)),
        PauliTestParam("XZ", PauliOperator("X 0", 2.0),
            PauliOperator("Z 0", 2.0), PauliOperator("Y 0", -4.0i)),
        PauliTestParam("YX", PauliOperator("Y 0", 2.0),
            PauliOperator("X 0", 2.0), PauliOperator("Z 0", -4.0i)),
        PauliTestParam("YY", PauliOperator("Y 0", 2.0),
            PauliOperator("Y 0", 2.0), PauliOperator("I 0", 4.0)),
        PauliTestParam("YZ", PauliOperator("Y 0", 2.0),
            PauliOperator("Z 0", 2.0), PauliOperator("X 0", 4.0i)),
        PauliTestParam("ZX", PauliOperator("Z 0", 2.0),
            PauliOperator("X 0", 2.0), PauliOperator("Y 0", 4.0i)),
        PauliTestParam("ZY", PauliOperator("Z 0", 2.0),
            PauliOperator("Y 0", 2.0), PauliOperator("X 0", -4.0i)),
        PauliTestParam("ZZ", PauliOperator("Z 0", 2.0),
            PauliOperator("Z 0", 2.0), PauliOperator("I 0", 4.0))),
    testing::PrintToStringParamName());
