#include <gtest/gtest.h>

#include <cppsim_experimental/fermion_operator.hpp>
#include <cppsim_experimental/observable.hpp>

template<typename Set>
bool set_compare(Set const &lhs, Set const &rhs){
    return lhs.size() == rhs.size()
        && equal(lhs.begin(), lhs.end(), rhs.begin());
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

    // TEST

    MultiQubitPauliOperator x_op("X 0");
    MultiQubitPauliOperator y_op("Y 0");
    std::string term_0_jw_exp = "(0.5+0j) [X 0 ] +\n(0+0.5j) [Y 0 ]";
    EXPECT_EQ(term_0_jw.get_term_count(), 2);
    for(int i = 0; i < term_0_jw.get_term_count(); i++) {
        auto term = term_0_jw.get_term(i);
        if(term.second == x_op)
            EXPECT_EQ(term.first, 0.5);
        else if(term.second == y_op)
            EXPECT_EQ(term.first, 0.5i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:" << term_0_jw_exp <<  "\nterm_0_jw:\n" << term_0_jw.to_string();
    }

    std::string term_0hat_jw_exp = "(0.5+0j) [X 0 ] +\n(0-0.5j) [Y 0 ]";
    EXPECT_EQ(term_0hat_jw.get_term_count(), 2);
    for(int i = 0; i < term_0hat_jw.get_term_count(); i++) {
        auto term = term_0hat_jw.get_term(i);
        if(term.second == x_op)
            EXPECT_EQ(term.first, 0.5);
        else if(term.second == y_op)
            EXPECT_EQ(term.first, -0.5i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:" << term_0hat_jw_exp << "\nterm_0hat_jw:\n" << term_0hat_jw.to_string();
    }

    std::string term_2_jw_exp =
        "(0.5+0j) [Z 0 Z 1 X 2 ] +\n"
        "(0+0.5j) [Z 0 Z 1 Y 2 ]";
    MultiQubitPauliOperator op1("Z 0 Z 1 X 2");
    MultiQubitPauliOperator op2("Z 0 Z 1 Y 2");
    EXPECT_EQ(term_2_jw.get_term_count(), 2);
    for(int i = 0; i < term_2_jw.get_term_count(); i++) {
        auto term = term_2_jw.get_term(i);
        if(term.second == op1)
            EXPECT_EQ(term.first, 0.5);
        else if(term.second == op2)
            EXPECT_EQ(term.first, 0.5i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:" << term_2_jw_exp << "\nterm_2_jw:\n" << term_2_jw.to_string();
    }

    std::string term_2hat_jw_exp = 
        "(0.5+0j) [Z 0 Z 1 X 2 ] +\n"
        "(0-0.5j) [Z 0 Z 1 Y 2 ]";
    EXPECT_EQ(term_2hat_jw.get_term_count(), 2);
    for(int i = 0; i < term_2hat_jw.get_term_count(); i++) {
        auto term = term_2hat_jw.get_term(i);
        if(term.second == op1)
            EXPECT_EQ(term.first, 0.5);
        else if(term.second == op2)
            EXPECT_EQ(term.first, -0.5i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:" << term_2hat_jw_exp << "\nterm_2hat_jw:\n" << term_2hat_jw.to_string();
    }
}

TEST(FermionOperatorTest, MultiActionJordanWignerTest1) {
    FermionOperator op1, op2;
    op1.add_term(1.0, "0 2^");
    op2.add_term(1.0, "0^ 2");

    Observable op1_jw = op1.jordan_wigner();
    Observable op2_jw = op2.jordan_wigner();

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
    for(int i = 0; i < op1_jw.get_term_count(); i++) {
        auto term = op1_jw.get_term(i);
        if(term.second == ex1)
            EXPECT_EQ(term.first, -0.25i);
        else if(term.second == ex2)
            EXPECT_EQ(term.first, -0.25);
        else if(term.second == ex3)
            EXPECT_EQ(term.first, -0.25);
        else if(term.second == ex4)
            EXPECT_EQ(term.first, 0.25i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n" << op1_expected << "\nop1_jw:\n" << op1_jw.to_string();
    }

    std::string op2_expected =
        "(0-0.25j) [Y 0 Z 1 X 2 ] +\n"
        "(0.25+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(0.25+0j) [X 0 Z 1 X 2 ] +\n"
        "(0+0.25j) [X 0 Z 1 Y 2 ]";
    EXPECT_EQ(op2_jw.get_term_count(), 4);
    for(int i = 0; i < op2_jw.get_term_count(); i++) {
        auto term = op2_jw.get_term(i);
        if(term.second == ex1)
            EXPECT_EQ(term.first, -0.25i);
        else if(term.second == ex2)
            EXPECT_EQ(term.first, 0.25);
        else if(term.second == ex3)
            EXPECT_EQ(term.first, 0.25);
        else if(term.second == ex4)
            EXPECT_EQ(term.first, 0.25i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n" << op2_expected << "\nop2_jw:\n" << op2_jw.to_string();
    }
    
    // TODO 係数が0の場合、Observableから削除する処理をする
    std::string op3_expected =
        "(0-0.5j) [Y 0 Z 1 X 2 ] +\n"
        "(0+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(0+0j) [X 0 Z 1 X 2 ] +\n"
        "(0+0.5j) [X 0 Z 1 Y 2 ]";
    EXPECT_EQ(op3.get_term_count(), 4);
    for(int i = 0; i < op3.get_term_count(); i++) {
        auto term = op3.get_term(i);
        if(term.second == ex1)
            EXPECT_EQ(term.first, -0.5i);
        else if(term.second == ex2)
            EXPECT_EQ(term.first, 0i);
        else if(term.second == ex3)
            EXPECT_EQ(term.first, 0i);
        else if(term.second == ex4)
            EXPECT_EQ(term.first, 0.5i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n" << op3_expected << "\nop3:\n" << op3.to_string();
    }
}

TEST(FermionOperatorTest, MultiActionJordanWignerTest2) {
    FermionOperator op1, op2;
    op1.add_term(1.0, "2^ 0");
    op2.add_term(1.0, "0^ 2");

    Observable op1_jw = op1.jordan_wigner();
    Observable op2_jw = op2.jordan_wigner();

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
    for(int i = 0; i < op1_jw.get_term_count(); i++) {
        auto term = op1_jw.get_term(i);
        if(term.second == ex1)
            EXPECT_EQ(term.first, 0.25i);
        else if(term.second == ex2)
            EXPECT_EQ(term.first, 0.25);
        else if(term.second == ex3)
            EXPECT_EQ(term.first, 0.25);
        else if(term.second == ex4)
            EXPECT_EQ(term.first, -0.25i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n" << op1_expected << "\nop1_jw:\n" << op1_jw.to_string();
    }

    std::string op2_expected =
        "(0+0.25j) [Y 0 Z 1 X 2 ] +\n"
        "(-0.25+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(-0.25+0j) [X 0 Z 1 X 2 ] +\n"
        "(0-0.25j) [X 0 Z 1 Y 2 ]";
    EXPECT_EQ(op2_jw.get_term_count(), 4);
    for(int i = 0; i < op2_jw.get_term_count(); i++) {
        auto term = op2_jw.get_term(i);
        if(term.second == ex1)
            EXPECT_EQ(term.first, -0.25i);
        else if(term.second == ex2)
            EXPECT_EQ(term.first, 0.25);
        else if(term.second == ex3)
            EXPECT_EQ(term.first, 0.25);
        else if(term.second == ex4)
            EXPECT_EQ(term.first, 0.25i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n" << op2_expected << "\nop2_jw:\n" << op2_jw.to_string();
    }
    // TODO 係数が0の場合、Observableから削除する処理をする
    std::string op3_expected =
        "(0+0j) [Y 0 Z 1 X 2 ] +\n"
        "(0.5+0j) [X 0 Z 1 X 2 ] +\n"
        "(0.5+0j) [Y 0 Z 1 Y 2 ] +\n"
        "(0+0j) [X 0 Z 1 Y 2 ]";

    EXPECT_EQ(op3.get_term_count(), 4);
    for(int i = 0; i < op3.get_term_count(); i++) {
        auto term = op3.get_term(i);
        if(term.second == ex1)
            EXPECT_EQ(term.first, 0i);
        else if(term.second == ex2)
            EXPECT_EQ(term.first, 0.5);
        else if(term.second == ex3)
            EXPECT_EQ(term.first, 0.5);
        else if(term.second == ex4)
            EXPECT_EQ(term.first, 0i);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n" << op3_expected << "\nop3:\n" << op3.to_string();
    }
}

TEST(FermionOperatorTest, count_qubitsTest){
    FermionOperator op1, op2, op3;
    op1.add_term(1.0, "0");
    op2.add_term(1.0, "0");
    op3.add_term(1.0, "3^ 2 1");

    EXPECT_EQ(count_qubits(op1), 1);
    EXPECT_EQ(count_qubits(op2), 1);
    EXPECT_EQ(count_qubits(op3), 4);
}

TEST(FermionOperatorTest, update_setTest){
    std::set<UINT> update_set_17 = _update_set(17,100);
    std::set<UINT> update_set_17_expected = {17, 19, 23, 31, 63};
    EXPECT_TRUE(set_compare(update_set_17, update_set_17_expected));
}

TEST(FermionOperatorTest, occupation_setTest){
    std::set<UINT> occupation_set_17 = _occupation_set(17);
    std::set<UINT> occupation_set_17_expected = {16, 17};
    EXPECT_TRUE(set_compare(occupation_set_17, occupation_set_17_expected));

    std::set<UINT> occupation_set_23 = _occupation_set(23);
    std::set<UINT> occupation_set_23_expected = {19, 21, 22, 23};
    EXPECT_TRUE(set_compare(occupation_set_23, occupation_set_23_expected));
}

TEST(FermionOperatorTest, parity_setTest){
    std::set<UINT> parity_set_16 = _parity_set(17 - 1);
    std::set<UINT> parity_set_16_expected = {15, 16};
    EXPECT_TRUE(set_compare(parity_set_16, parity_set_16_expected));

    std::set<UINT> parity_set_30 = _parity_set(30);
    std::set<UINT> parity_set_30_expected = {15, 23, 27, 29, 30};
    EXPECT_TRUE(set_compare(parity_set_30, parity_set_30_expected));
}

TEST(FermionOperatorTest, bravyi_kitaevTest1){
    FermionOperator op1;
    op1.add_term(1.0, "17");
    op1.add_term(1.0, "17^");

    Observable res = op1.bravyi_kitaev(100);

    MultiQubitPauliOperator ex1("Z 15 Y 17 X 19 X 23 X 31 X 63");
    MultiQubitPauliOperator ex2("Z 15 Z 16 X 17 X 19 X 23 X 31 X 63");

    std::string res_expected =
    "(0+0j) [Z 15 Y 17 X 19 X 23 X 31 X 63 ] +\n"
    "(1+0j) [Z 15 Z 16 X 17 X 19 X 23 X 31 X 63 ]";

    EXPECT_EQ(res.get_term_count(), 2);
    for(int i = 0; i < res.get_term_count(); i++) {
        auto term = res.get_term(i);
        if(term.second == ex1)
            EXPECT_EQ(term.first, 0i);
        else if(term.second == ex2)
            EXPECT_EQ(term.first, 1.0);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n" << res_expected << "\nres:\n" << res.to_string();
    }
}

TEST(FermionOperatorTest, bravyi_kitaevTest2){
    FermionOperator op1;
    op1.add_term(1.0, "50");
    op1.add_term(1.0, "50^");

    Observable res = op1.bravyi_kitaev(100);

    MultiQubitPauliOperator ex1("Z 31 Z 47 Z 49 Y 50 X 51 X 55 X 63");
    MultiQubitPauliOperator ex2("Z 31 Z 47 Z 49 X 50 X 51 X 55 X 63");

    std::string res_expected =
        "(0+0j) [Z 31 Z 47 Z 49 Y 50 X 51 X 55 X 63 ] +\n"
        "(1+0j) [Z 31 Z 47 Z 49 X 50 X 51 X 55 X 63 ]";

    EXPECT_EQ(res.get_term_count(), 2);
    for(int i = 0; i < res.get_term_count(); i++) {
        auto term = res.get_term(i);
        if(term.second == ex1)
            EXPECT_EQ(term.first, 0i);
        else if(term.second == ex2)
            EXPECT_EQ(term.first, 1.0);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n" << res_expected << "\nres:\n" << res.to_string();
    }
}

TEST(FermionOperatorTest, bravyi_kitaevTest3){
    FermionOperator op1;
    op1.add_term(1.0, "73");
    op1.add_term(1.0, "73^");

    Observable res = op1.bravyi_kitaev(100);

    MultiQubitPauliOperator ex1("Z 63 Z 71 Y 73 X 75 X 79 X 95");
    MultiQubitPauliOperator ex2("Z 63 Z 71 Z 72 X 73 X 75 X 79 X 95");

    std::string res_expected =
        "(0+0j) [Z 63 Z 71 Y 73 X 75 X 79 X 95 ] +\n"
        "(1+0j) [Z 63 Z 71 Z 72 X 73 X 75 X 79 X 95 ]";

    EXPECT_EQ(res.get_term_count(), 2);
    for(int i = 0; i < res.get_term_count(); i++) {
        auto term = res.get_term(i);
        if(term.second == ex1)
            EXPECT_EQ(term.first, 0i);
        else if(term.second == ex2)
            EXPECT_EQ(term.first, 1.0);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n" << res_expected << "\nres:\n" << res.to_string();
    }
}