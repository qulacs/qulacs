#include <gtest/gtest.h>

#include <cppsim/pauli_operator.hpp>
#include <cppsim/state.hpp>

TEST(PauliOperatorTest, ContainsExtraWhitespace) {
    PauliOperator expected = PauliOperator("X 0", 1.0);
    PauliOperator pauli_whitespace = PauliOperator("X 0 ", 1.0);

    EXPECT_EQ(1, pauli_whitespace.get_index_list().size());
    EXPECT_EQ(1, pauli_whitespace.get_pauli_id_list().size());
    EXPECT_EQ(expected.get_pauli_string(), pauli_whitespace.get_pauli_string());
}

TEST(PauliOperatorTest, EmptyStringConstructsIdentity) {
    const auto identity = PauliOperator("", 1.0);
    ASSERT_EQ(0, identity.get_index_list().size());
    ASSERT_EQ(0, identity.get_pauli_id_list().size());
    ASSERT_EQ("", identity.get_pauli_string());
}

TEST(PauliOperatorTest, PauliQubitOverflow) {
    int n = 2;
    double coef = 2.0;
    std::string Pauli_string = "X 0 X 1 X 3";
    PauliOperator pauli = PauliOperator(Pauli_string, coef);
    QuantumState state = QuantumState(n);
    state.set_Haar_random_state();
    EXPECT_THROW(
        pauli.get_expectation_value(&state), InvalidPauliIdentifierException);
}

TEST(PauliOperatorTest, BrokenPauliStringA) {
    double coef = 2.0;
    std::string Pauli_string = "X 0 X Z 1 Y 2";
    EXPECT_THROW(
        PauliOperator(Pauli_string, coef), InvalidPauliIdentifierException);
}

TEST(PauliOperatorTest, BrokenPauliStringB) {
    double coef = 2.0;
    std::string Pauli_string = "X {i}";
    EXPECT_THROW(
        PauliOperator(Pauli_string, coef), InvalidPauliIdentifierException);
}

TEST(PauliOperatorTest, BrokenPauliStringC) {
    double coef = 2.0;
    std::string Pauli_string = "X 4x";
    EXPECT_THROW(
        PauliOperator(Pauli_string, coef), InvalidPauliIdentifierException);
}

TEST(PauliOperatorTest, BrokenPauliStringD) {
    double coef = 2.0;
    std::string Pauli_string = "4 X";
    EXPECT_THROW(
        PauliOperator(Pauli_string, coef), InvalidPauliIdentifierException);
}

TEST(PauliOperatorTest, SpacedPauliString) {
    double coef = 2.0;
    std::string Pauli_string = "X 0 Y 1 ";
    PauliOperator pauli = PauliOperator(Pauli_string, coef);
    size_t PauliSize = pauli.get_index_list().size();
    ASSERT_EQ(PauliSize, 2);
}

TEST(PauliOperatorTest, PartedPauliString) {
    double coef = 2.0;
    std::string Pauli_string = "X 0 Y ";
    EXPECT_THROW(
        PauliOperator(Pauli_string, coef), InvalidPauliIdentifierException);
}

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

INSTANTIATE_TEST_CASE_P(MultiPauli, PauliOperatorMultiplyTest,
    testing::Values(
        PauliTestParam("X_Y", PauliOperator("X 0", 2.0),
            PauliOperator("Y 1", 2.0), PauliOperator("X 0 Y 1", 4.0)),
        PauliTestParam("XY_YX", PauliOperator("X 0 Y 1", 2.0),
            PauliOperator("Y 0 X 1", 2.0), PauliOperator("Z 0 Z 1", 4.0))),
    testing::PrintToStringParamName());

// OpenMPによって並列化した場合でも、計算結果のindexの順序が保たれることを確認する
INSTANTIATE_TEST_CASE_P(
    MultiPauliWithOpenMP, PauliOperatorMultiplyTest,
    []() {
        double coef = 2.0;
        unsigned int MAX_TERM = 100;
        std::string pauli_string_x = "";
        std::string pauli_string_y = "";
        std::string pauli_string_z = "";

        for (int i = 0; i < MAX_TERM; i++) {
            pauli_string_x += "X " + std::to_string(i);
            pauli_string_y += "Y " + std::to_string(i);
            pauli_string_z += "Z " + std::to_string(i);
            if (i + 1 < MAX_TERM) {
                pauli_string_x += " ";
                pauli_string_y += " ";
                pauli_string_z += " ";
            }
        }

        PauliOperator expected = PauliOperator(pauli_string_x, coef * coef);
        PauliOperator pauli_y = PauliOperator(pauli_string_y, coef);
        PauliOperator pauli_z = PauliOperator(pauli_string_z, coef);

        return testing::Values(
            PauliTestParam("Z_Y", pauli_z, pauli_y, expected));
    }(),
    testing::PrintToStringParamName());
