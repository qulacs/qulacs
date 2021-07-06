#include <gtest/gtest.h>

#include <cppsim_experimental/fermion_operator.hpp>
#include <cppsim_experimental/observable.hpp>

TEST(FermionOperatorTest, GetTermCountTest) {
    FermionOperator fermion_operator;
    fermion_operator.add_term(1.0, "2^ 1");
    EXPECT_EQ(1, fermion_operator.get_term_count());

    fermion_operator.add_term(2.0, "5^ 4 3^");
    EXPECT_EQ(2, fermion_operator.get_term_count());
}

TEST(FermionOperatorTest, AddGetTermTest) {
    FermionOperator fermion_operator;
    fermion_operator.add_term(1.0, "2^ 1");
    fermion_operator.add_term(-2.0, "5^ 4 3^");
    SingleFermionOperator op("5^ 4 3^");
    fermion_operator.add_term(-2.0, op);

    // AddしたものとGetしたものが一致することを確認する
    auto term1 = fermion_operator.get_term(0);
    EXPECT_EQ(1.0, term1.first);
    EXPECT_EQ(2, term1.second.get_target_index_list().at(0));
    EXPECT_EQ(ACTION_CREATE_ID, term1.second.get_action_id_list().at(0));
    EXPECT_EQ(1, term1.second.get_target_index_list().at(1));
    EXPECT_EQ(ACTION_DESTROY_ID, term1.second.get_action_id_list().at(1));

    auto term2 = fermion_operator.get_term(1);
    EXPECT_EQ(-2.0, term2.first);
    EXPECT_EQ(5, term2.second.get_target_index_list().at(0));
    EXPECT_EQ(ACTION_CREATE_ID, term2.second.get_action_id_list().at(0));
    EXPECT_EQ(4, term2.second.get_target_index_list().at(1));
    EXPECT_EQ(ACTION_DESTROY_ID, term2.second.get_action_id_list().at(1));
    EXPECT_EQ(3, term2.second.get_target_index_list().at(2));
    EXPECT_EQ(ACTION_CREATE_ID, term2.second.get_action_id_list().at(2));

    // SingleFermionOperatorを用いてAddした場合と、文字列を用いてAddした場合で同じ結果
    // が得られることを確認する
    auto term3 = fermion_operator.get_term(2);
    EXPECT_EQ(term3.first, term2.first);
    EXPECT_EQ(term3.second.get_target_index_list().at(0),
        term2.second.get_target_index_list().at(0));
    EXPECT_EQ(term3.second.get_action_id_list().at(0),
        term2.second.get_action_id_list().at(0));
    EXPECT_EQ(term3.second.get_target_index_list().at(1),
        term2.second.get_target_index_list().at(1));
    EXPECT_EQ(term3.second.get_action_id_list().at(1),
        term2.second.get_action_id_list().at(1));
    EXPECT_EQ(term3.second.get_target_index_list().at(2),
        term2.second.get_target_index_list().at(2));
    EXPECT_EQ(term3.second.get_action_id_list().at(2),
        term2.second.get_action_id_list().at(2));
}

TEST(FermionOperatorTest, RemoveTermTest) {
    FermionOperator fermion_operator;
    fermion_operator.add_term(1.0, "2^ 1");
    fermion_operator.add_term(-2.0, "5^ 4 3^");
    fermion_operator.add_term(3.0, "6 7^");

    EXPECT_EQ(3, fermion_operator.get_term_count());

    // Removeした結果、Termは1個減る
    fermion_operator.remove_term(1);
    EXPECT_EQ(2, fermion_operator.get_term_count());

    auto term = fermion_operator.get_term(1);
    EXPECT_EQ(3.0, term.first);
    EXPECT_EQ(6, term.second.get_target_index_list().at(0));
    EXPECT_EQ(ACTION_DESTROY_ID, term.second.get_action_id_list().at(0));
    EXPECT_EQ(7, term.second.get_target_index_list().at(1));
    EXPECT_EQ(ACTION_CREATE_ID, term.second.get_action_id_list().at(1));
}

TEST(FermionOperatorTest, GetFermionListTest) {
    FermionOperator fermion_operator;
    fermion_operator.add_term(1.0, "2^ 1");
    fermion_operator.add_term(2.0, "5^ 4 3^");

    auto sfop_list = fermion_operator.get_fermion_list();
    EXPECT_EQ(2, sfop_list.size());
    auto target_index_list = sfop_list.at(0).get_target_index_list();
    EXPECT_EQ(2, target_index_list.at(0));
    EXPECT_EQ(1, target_index_list.at(1));
}

TEST(FermionOperatorTest, GetCoefListTest) {
    FermionOperator fermion_operator;
    fermion_operator.add_term(1.0, "2^ 1");
    fermion_operator.add_term(2.0, "5^ 4 3^");

    auto sfop_list = fermion_operator.get_coef_list();
    EXPECT_EQ(2, sfop_list.size());
    EXPECT_EQ(1.0, sfop_list.at(0));
    EXPECT_EQ(2.0, sfop_list.at(1));
}

TEST(FermionOperatorTest, SingleActionJordanWignerTest) {
    FermionOperator term_0, term_0hat;
    FermionOperator term_2, term_2hat;

    term_0.add_term(1.0, "0");
    term_0hat.add_term(1.0, "0^");
    term_2.add_term(1.0, "2");
    term_2hat.add_term(1.0, "2^");

    Observable term_0_jw = term_0.jordan_wigner();
    Observable term_0hat_jw = term_0hat.jordan_wigner();
    Observable term_2_jw = term_2.jordan_wigner();
    Observable term_2hat_jw = term_2hat.jordan_wigner();

    std::string term_0_jw_expected = "(0.5+0j) [X 0 ] +\n(0+0.5j) [Y 0 ]";
    std::string term_0hat_jw_expected = "(0.5+0j) [X 0 ] +\n(0-0.5j) [Y 0 ]";
    std::string term_2_jw_expected =
        "(0.5+0j) [Z 0 Z 1 X 2 ] +\n"
        "(0+0.5j) [Z 0 Z 1 Y 2 ]";
    std::string term_2hat_jw_expected =
        "(0.5+0j) [Z 0 Z 1 X 2 ] +\n"
        "(0-0.5j) [Z 0 Z 1 Y 2 ]";
    EXPECT_EQ(term_0_jw.to_string(), term_0_jw_expected);
    EXPECT_EQ(term_0hat_jw.to_string(), term_0hat_jw_expected);
    EXPECT_EQ(term_2_jw.to_string(), term_2_jw_expected);
    EXPECT_EQ(term_2hat_jw.to_string(), term_2hat_jw_expected);
}

TEST(FermionOperatorTest, MultiActionJordanWignerTest1) {
    FermionOperator op1, op2;
    op1.add_term(1.0, "0 2^");
    op2.add_term(1.0, "0^ 2");

    Observable op1_jw = op1.jordan_wigner();
    Observable op2_jw = op2.jordan_wigner();

    Observable op3 = op1_jw + op2_jw;

    std::string op1_expected =
        "(0-0.25j) [Y 0 Z 1 X 2 ] +\n"
        "(-0.25-0j) [Y 0 Z 1 Y 2 ] +\n"
        "(-0.25+0j) [X 0 Z 1 X 2 ] +\n"
        "(0+0.25j) [X 0 Z 1 Y 2 ]";
    std::string op2_expected =
        "(0-0.25j) [Y 0 Z 1 X 2 ] +\n"
        "(0.25+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(0.25+0j) [X 0 Z 1 X 2 ] +\n"
        "(0+0.25j) [X 0 Z 1 Y 2 ]";
    // TODO 係数が0の場合、Observableから削除する処理をする
    std::string op3_expected =
        "(0-0.5j) [Y 0 Z 1 X 2 ] +\n"
        "(0+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(0+0j) [X 0 Z 1 X 2 ] +\n"
        "(0+0.5j) [X 0 Z 1 Y 2 ]";

    EXPECT_EQ(op1_jw.to_string(), op1_expected);
    EXPECT_EQ(op2_jw.to_string(), op2_expected);
    EXPECT_EQ(op3.to_string(), op3_expected);
}

TEST(FermionOperatorTest, MultiActionJordanWignerTest2) {
    FermionOperator op1, op2;
    op1.add_term(1.0, "2^ 0");
    op2.add_term(1.0, "0^ 2");

    Observable op1_jw = op1.jordan_wigner();
    Observable op2_jw = op2.jordan_wigner();

    Observable op3 = op1_jw + op2_jw;

    std::string op1_expected =
        "(0-0.25j) [Y 0 Z 1 X 2 ] +\n"
        "(-0.25-0j) [Y 0 Z 1 Y 2 ] +\n"
        "(-0.25+0j) [X 0 Z 1 X 2 ] +\n"
        "(0+0.25j) [X 0 Z 1 Y 2 ]";
    std::string op2_expected =
        "(0-0.25j) [Y 0 Z 1 X 2 ] +\n"
        "(0.25+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(0.25+0j) [X 0 Z 1 X 2 ] +\n"
        "(0+0.25j) [X 0 Z 1 Y 2 ]";
    // TODO 係数が0の場合、Observableから削除する処理をする
    std::string op3_expected =
        "(0+0j) [Y 0 Z 1 X 2 ] +\n"
        "(0.5+0j) [X 0 Z 1 X 2 ] +\n"
        "(0.5+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(0+0j) [X 0 Z 1 Y 2 ]";

    EXPECT_EQ(op3.to_string(), op3_expected);
}