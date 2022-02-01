#include <gtest/gtest.h>

#include <cppsim/pauli_operator.hpp>

TEST(PauliOperatorTest, ContainsExtraWhitespace) {
    PauliOperator expected = PauliOperator("X 0", 1.0);
    PauliOperator pauli_whitespace = PauliOperator("X 0 ", 1.0);

    EXPECT_EQ(1, pauli_whitespace.get_index_list().size());
    EXPECT_EQ(1, pauli_whitespace.get_pauli_id_list().size());
    EXPECT_EQ(expected.get_pauli_string(), pauli_whitespace.get_pauli_string());
}
