#include <gtest/gtest.h>

#include <cppsim_experimental/fermion_operator.hpp>
#include <cppsim_experimental/jordan_wigner.hpp>

TEST(JordanWignerTest, SingleActionJordanWignerTest) {
    FermionOperator term_0, term_0hat;
    FermionOperator term_2, term_2hat;

    term_0.add_term(1.0, "0");
    term_0hat.add_term(1.0, "0^");
    term_2.add_term(1.0, "2");
    term_2hat.add_term(1.0, "2^");

    Observable term_0_jw = transforms::jordan_wigner(term_0);
    Observable term_0hat_jw = transforms::jordan_wigner(term_0hat);
    Observable term_2_jw = transforms::jordan_wigner(term_2);
    Observable term_2hat_jw = transforms::jordan_wigner(term_2hat);

    // TEST

    MultiQubitPauliOperator x_op("X 0");
    MultiQubitPauliOperator y_op("Y 0");
    std::string term_0_jw_exp = "(0.5+0j) [X 0 ] +\n(0+0.5j) [Y 0 ]";
    EXPECT_EQ(term_0_jw.get_term_count(), 2);
    for (int i = 0; i < term_0_jw.get_term_count(); i++) {
        auto term = term_0_jw.get_term(i);
        if (term.second == x_op)
            EXPECT_EQ(term.first, 0.5);
        else if (term.second == y_op)
            EXPECT_EQ(term.first, 0.5i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:" << term_0_jw_exp
                          << "\nterm_0_jw:\n"
                          << term_0_jw.to_string();
    }

    std::string term_0hat_jw_exp = "(0.5+0j) [X 0 ] +\n(0-0.5j) [Y 0 ]";
    EXPECT_EQ(term_0hat_jw.get_term_count(), 2);
    for (int i = 0; i < term_0hat_jw.get_term_count(); i++) {
        auto term = term_0hat_jw.get_term(i);
        if (term.second == x_op)
            EXPECT_EQ(term.first, 0.5);
        else if (term.second == y_op)
            EXPECT_EQ(term.first, -0.5i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:" << term_0hat_jw_exp
                          << "\nterm_0hat_jw:\n"
                          << term_0hat_jw.to_string();
    }

    std::string term_2_jw_exp =
        "(0.5+0j) [Z 0 Z 1 X 2 ] +\n"
        "(0+0.5j) [Z 0 Z 1 Y 2 ]";
    MultiQubitPauliOperator op1("Z 0 Z 1 X 2");
    MultiQubitPauliOperator op2("Z 0 Z 1 Y 2");
    EXPECT_EQ(term_2_jw.get_term_count(), 2);
    for (int i = 0; i < term_2_jw.get_term_count(); i++) {
        auto term = term_2_jw.get_term(i);
        if (term.second == op1)
            EXPECT_EQ(term.first, 0.5);
        else if (term.second == op2)
            EXPECT_EQ(term.first, 0.5i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:" << term_2_jw_exp
                          << "\nterm_2_jw:\n"
                          << term_2_jw.to_string();
    }

    std::string term_2hat_jw_exp =
        "(0.5+0j) [Z 0 Z 1 X 2 ] +\n"
        "(0-0.5j) [Z 0 Z 1 Y 2 ]";
    EXPECT_EQ(term_2hat_jw.get_term_count(), 2);
    for (int i = 0; i < term_2hat_jw.get_term_count(); i++) {
        auto term = term_2hat_jw.get_term(i);
        if (term.second == op1)
            EXPECT_EQ(term.first, 0.5);
        else if (term.second == op2)
            EXPECT_EQ(term.first, -0.5i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:" << term_2hat_jw_exp
                          << "\nterm_2hat_jw:\n"
                          << term_2hat_jw.to_string();
    }
}

TEST(JordanWignerTest, MultiActionJordanWignerTest1) {
    FermionOperator op1, op2;
    op1.add_term(1.0, "0 2^");
    op2.add_term(1.0, "0^ 2");

    Observable op1_jw = transforms::jordan_wigner(op1);
    Observable op2_jw = transforms::jordan_wigner(op2);

    Observable op3 = op1_jw + op2_jw;

    // TEST

    MultiQubitPauliOperator ex1("Y 0 Z 1 X 2");
    MultiQubitPauliOperator ex2("Y 0 Z 1 Y 2");
    MultiQubitPauliOperator ex3("X 0 Z 1 X 2");
    MultiQubitPauliOperator ex4("X 0 Z 1 Y 2");

    std::string op1_expected =
        "(0-0.25j) [Y 0 Z 1 X 2 ] +\n"
        "(-0.25-0j) [Y 0 Z 1 Y 2 ] +\n"
        "(-0.25+0j) [X 0 Z 1 X 2 ] +\n"
        "(0+0.25j) [X 0 Z 1 Y 2 ]";
    EXPECT_EQ(op1_jw.get_term_count(), 4);
    for (int i = 0; i < op1_jw.get_term_count(); i++) {
        auto term = op1_jw.get_term(i);
        if (term.second == ex1)
            EXPECT_EQ(term.first, -0.25i);
        else if (term.second == ex2)
            EXPECT_EQ(term.first, -0.25);
        else if (term.second == ex3)
            EXPECT_EQ(term.first, -0.25);
        else if (term.second == ex4)
            EXPECT_EQ(term.first, 0.25i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n"
                          << op1_expected << "\nop1_jw:\n"
                          << op1_jw.to_string();
    }

    std::string op2_expected =
        "(0-0.25j) [Y 0 Z 1 X 2 ] +\n"
        "(0.25+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(0.25+0j) [X 0 Z 1 X 2 ] +\n"
        "(0+0.25j) [X 0 Z 1 Y 2 ]";
    EXPECT_EQ(op2_jw.get_term_count(), 4);
    for (int i = 0; i < op2_jw.get_term_count(); i++) {
        auto term = op2_jw.get_term(i);
        if (term.second == ex1)
            EXPECT_EQ(term.first, -0.25i);
        else if (term.second == ex2)
            EXPECT_EQ(term.first, 0.25);
        else if (term.second == ex3)
            EXPECT_EQ(term.first, 0.25);
        else if (term.second == ex4)
            EXPECT_EQ(term.first, 0.25i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n"
                          << op2_expected << "\nop2_jw:\n"
                          << op2_jw.to_string();
    }

    // TODO 係数が0の場合、Observableから削除する処理をする
    std::string op3_expected =
        "(0-0.5j) [Y 0 Z 1 X 2 ] +\n"
        "(0+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(0+0j) [X 0 Z 1 X 2 ] +\n"
        "(0+0.5j) [X 0 Z 1 Y 2 ]";
    EXPECT_EQ(op3.get_term_count(), 4);
    for (int i = 0; i < op3.get_term_count(); i++) {
        auto term = op3.get_term(i);
        if (term.second == ex1)
            EXPECT_EQ(term.first, -0.5i);
        else if (term.second == ex2)
            EXPECT_EQ(term.first, 0i);
        else if (term.second == ex3)
            EXPECT_EQ(term.first, 0i);
        else if (term.second == ex4)
            EXPECT_EQ(term.first, 0.5i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n"
                          << op3_expected << "\nop3:\n"
                          << op3.to_string();
    }
}

TEST(JordanWignerTest, MultiActionJordanWignerTest2) {
    FermionOperator op1, op2;
    op1.add_term(1.0, "2^ 0");
    op2.add_term(1.0, "0^ 2");

    Observable op1_jw = transforms::jordan_wigner(op1);
    Observable op2_jw = transforms::jordan_wigner(op2);

    Observable op3 = op1_jw + op2_jw;

    // TEST

    MultiQubitPauliOperator ex1("Y 0 Z 1 X 2");
    MultiQubitPauliOperator ex2("Y 0 Z 1 Y 2");
    MultiQubitPauliOperator ex3("X 0 Z 1 X 2");
    MultiQubitPauliOperator ex4("X 0 Z 1 Y 2");

    std::string op1_expected =
        "(0+0.25j) [Y 0 Z 1 X 2 ] +\n"
        "(0.25+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(0.25+0j) [X 0 Z 1 X 2 ] +\n"
        "(0-0.25j) [X 0 Z 1 Y 2 ]";
    EXPECT_EQ(op1_jw.get_term_count(), 4);
    for (int i = 0; i < op1_jw.get_term_count(); i++) {
        auto term = op1_jw.get_term(i);
        if (term.second == ex1)
            EXPECT_EQ(term.first, 0.25i);
        else if (term.second == ex2)
            EXPECT_EQ(term.first, 0.25);
        else if (term.second == ex3)
            EXPECT_EQ(term.first, 0.25);
        else if (term.second == ex4)
            EXPECT_EQ(term.first, -0.25i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n"
                          << op1_expected << "\nop1_jw:\n"
                          << op1_jw.to_string();
    }

    std::string op2_expected =
        "(0+0.25j) [Y 0 Z 1 X 2 ] +\n"
        "(-0.25+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(-0.25+0j) [X 0 Z 1 X 2 ] +\n"
        "(0-0.25j) [X 0 Z 1 Y 2 ]";
    EXPECT_EQ(op2_jw.get_term_count(), 4);
    for (int i = 0; i < op2_jw.get_term_count(); i++) {
        auto term = op2_jw.get_term(i);
        if (term.second == ex1)
            EXPECT_EQ(term.first, -0.25i);
        else if (term.second == ex2)
            EXPECT_EQ(term.first, 0.25);
        else if (term.second == ex3)
            EXPECT_EQ(term.first, 0.25);
        else if (term.second == ex4)
            EXPECT_EQ(term.first, 0.25i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n"
                          << op2_expected << "\nop2_jw:\n"
                          << op2_jw.to_string();
    }
    // TODO 係数が0の場合、Observableから削除する処理をする
    std::string op3_expected =
        "(0+0j) [Y 0 Z 1 X 2 ] +\n"
        "(0.5+0j) [X 0 Z 1 X 2 ] +\n"
        "(0.5+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(0+0j) [X 0 Z 1 Y 2 ]";
    EXPECT_EQ(op3.get_term_count(), 4);
    for (int i = 0; i < op3.get_term_count(); i++) {
        auto term = op3.get_term(i);
        if (term.second == ex1)
            EXPECT_EQ(term.first, 0i);
        else if (term.second == ex2)
            EXPECT_EQ(term.first, 0.5);
        else if (term.second == ex3)
            EXPECT_EQ(term.first, 0.5);
        else if (term.second == ex4)
            EXPECT_EQ(term.first, 0i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n"
                          << op3_expected << "\nop3:\n"
                          << op3.to_string();
    }
}