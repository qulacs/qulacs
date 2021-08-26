#include <gtest/gtest.h>

#include <cppsim_experimental/single_fermion_operator.hpp>

TEST(SingleFermionOperatorTest, InitTest){
    // 異なる初期化方法で同じ結果が得られることをテストする
    std::vector<UINT> target_index{4, 3, 2, 1};
    std::vector<UINT> action_id{1, 0, 1, 0};
    std::string action_string = "4^ 3 2^ 1";

    SingleFermionOperator op1(target_index, action_id);
    SingleFermionOperator op2(action_string);

    ASSERT_EQ(4, op1.get_target_index_list().size());
    ASSERT_EQ(4, op2.get_target_index_list().size());
    ASSERT_EQ(4, op1.get_action_id_list().size());
    ASSERT_EQ(4, op2.get_action_id_list().size());

    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(op1.get_target_index_list().at(i), op2.get_target_index_list().at(i));
        EXPECT_EQ(op1.get_action_id_list().at(i), op2.get_action_id_list().at(i));
    }
}

TEST(SingleFermionOperatorTest, to_stringTest){
    std::string action_string = "4^ 3 2^ 1";
    SingleFermionOperator op(action_string);

    EXPECT_EQ(op.to_string(), action_string);
}

TEST(SingleFermionOperatorTest, operatorTest){
    SingleFermionOperator expected("3^ 1 2^ 1");
    SingleFermionOperator op1("3^ 1");
    SingleFermionOperator op2("2^ 1");
    SingleFermionOperator op3 = op1*op2;

    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(op3.get_target_index_list().at(i), expected.get_target_index_list().at(i));
        EXPECT_EQ(op3.get_action_id_list().at(i), expected.get_action_id_list().at(i));
    }

    op1 *= op2;

    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(op1.get_target_index_list().at(i), expected.get_target_index_list().at(i));
        EXPECT_EQ(op1.get_action_id_list().at(i), expected.get_action_id_list().at(i));
    }
}