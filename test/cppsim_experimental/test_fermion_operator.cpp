#include <gtest/gtest.h>

#include <cppsim_experimental/fermion_operator.hpp>
#include <cppsim_experimental/observable.hpp>

template <typename Set>
bool set_compare(Set const &lhs, Set const &rhs) {
    return lhs.size() == rhs.size() &&
           equal(lhs.begin(), lhs.end(), rhs.begin());
}

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

TEST(FermionOperatorTest, add_operatorTest) {
    FermionOperator op1;
    op1.add_term(1.0, "2^ 1");
    op1.add_term(2.0, "3^ 2");
    FermionOperator op2;
    op2.add_term(2.0, "2^ 1");
    op2.add_term(3.0, "4^ 3");

    FermionOperator expected;
    expected.add_term(3.0, "2^ 1");
    expected.add_term(2.0, "3^ 2");
    expected.add_term(3.0, "4^ 3");

    FermionOperator res = op1 + op2;

    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(res.get_term(i).first, expected.get_term(i).first);
        EXPECT_EQ(res.get_term(i).second.to_string(),
            expected.get_term(i).second.to_string());
    }

    op1 += op2;
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(op1.get_term(i).first, expected.get_term(i).first);
        EXPECT_EQ(op1.get_term(i).second.to_string(),
            expected.get_term(i).second.to_string());
    }
}

TEST(FermionOperatorTest, sub_operatorTest) {
    FermionOperator op1;
    op1.add_term(1.0, "2^ 1");
    op1.add_term(2.0, "3^ 2");
    FermionOperator op2;
    op2.add_term(2.0, "2^ 1");
    op2.add_term(3.0, "4^ 3");

    FermionOperator expected;
    expected.add_term(-1.0, "2^ 1");
    expected.add_term(2.0, "3^ 2");
    expected.add_term(-3.0, "4^ 3");

    FermionOperator res = op1 - op2;

    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(res.get_term(i).first, expected.get_term(i).first);
        EXPECT_EQ(res.get_term(i).second.to_string(),
            expected.get_term(i).second.to_string());
    }

    op1 -= op2;
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(op1.get_term(i).first, expected.get_term(i).first);
        EXPECT_EQ(op1.get_term(i).second.to_string(),
            expected.get_term(i).second.to_string());
    }
}

TEST(FermionOperatorTest, mul_operatorTest) {
    FermionOperator op1;
    op1.add_term(1.0, "2^ 1");
    op1.add_term(2.0, "3^ 2");
    FermionOperator op2;
    op2.add_term(2.0, "2^ 1");
    op2.add_term(3.0, "4^ 3");

    FermionOperator expected;
    expected.add_term(2.0, "2^ 1 2^ 1");
    expected.add_term(3.0, "2^ 1 4^ 3");
    expected.add_term(4.0, "3^ 2 2^ 1");
    expected.add_term(6.0, "3^ 2 4^ 3");

    FermionOperator res = op1 * op2;

    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(res.get_term(i).first, expected.get_term(i).first);
        EXPECT_EQ(res.get_term(i).second.to_string(),
            expected.get_term(i).second.to_string());
    }

    op1 *= op2;
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(op1.get_term(i).first, expected.get_term(i).first);
        EXPECT_EQ(op1.get_term(i).second.to_string(),
            expected.get_term(i).second.to_string());
    }
}