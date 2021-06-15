#include <gtest/gtest.h>

#include <cppsim_experimental/fermion_operator.hpp>
#include <cppsim_experimental/observable.hpp>

TEST(FermionOperatorTest, GetTermCountTest){
    FermionOperator fermion_operator;
    fermion_operator.add_term(1.0, "2^ 1");
    EXPECT_EQ(1, fermion_operator.get_term_count());

    fermion_operator.add_term(2.0, "5^ 4 3^");
    EXPECT_EQ(2, fermion_operator.get_term_count());
}

TEST(FermionOperatorTest, AddGetTermTest){
    FermionOperator fermion_operator;
    fermion_operator.add_term(1.0, "2^ 1");
    fermion_operator.add_term(-2.0, "5^ 4 3^");

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
}

