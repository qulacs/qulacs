#include <gtest/gtest.h>

#include <cppsim_experimental/bravyi_kitaev.hpp>
#include <cppsim_experimental/fermion_operator.hpp>

using namespace transforms;

template <typename Set>
bool set_compare(Set const &lhs, Set const &rhs) {
    return lhs.size() == rhs.size() &&
           equal(lhs.begin(), lhs.end(), rhs.begin());
}

TEST(BravyiKitaevTest, count_qubitsTest) {
    FermionOperator op1, op2, op3;
    op1.add_term(1.0, "0");
    op2.add_term(1.0, "0");
    op3.add_term(1.0, "3^ 2 1");

    EXPECT_EQ(count_qubits(op1), 1);
    EXPECT_EQ(count_qubits(op2), 1);
    EXPECT_EQ(count_qubits(op3), 4);
}

TEST(BravyiKitaevTest, update_setTest) {
    std::set<UINT> update_set_17 = _update_set(17, 100);
    std::set<UINT> update_set_17_expected = {17, 19, 23, 31, 63};
    EXPECT_TRUE(set_compare(update_set_17, update_set_17_expected));
}

TEST(BravyiKitaevTest, occupation_setTest) {
    std::set<UINT> occupation_set_17 = _occupation_set(17);
    std::set<UINT> occupation_set_17_expected = {16, 17};
    EXPECT_TRUE(set_compare(occupation_set_17, occupation_set_17_expected));

    std::set<UINT> occupation_set_23 = _occupation_set(23);
    std::set<UINT> occupation_set_23_expected = {19, 21, 22, 23};
    EXPECT_TRUE(set_compare(occupation_set_23, occupation_set_23_expected));
}

TEST(BravyiKitaevTest, parity_setTest) {
    std::set<UINT> parity_set_16 = _parity_set(17 - 1);
    std::set<UINT> parity_set_16_expected = {15, 16};
    EXPECT_TRUE(set_compare(parity_set_16, parity_set_16_expected));

    std::set<UINT> parity_set_30 = _parity_set(30);
    std::set<UINT> parity_set_30_expected = {15, 23, 27, 29, 30};
    EXPECT_TRUE(set_compare(parity_set_30, parity_set_30_expected));
}

TEST(BravyiKitaevTest, bravyi_kitaevTest1) {
    FermionOperator op1;
    op1.add_term(1.0, "17");
    op1.add_term(1.0, "17^");

    Observable res = bravyi_kitaev(op1, 100);

    MultiQubitPauliOperator ex1("Z 15 Y 17 X 19 X 23 X 31 X 63");
    MultiQubitPauliOperator ex2("Z 15 Z 16 X 17 X 19 X 23 X 31 X 63");

    std::string res_expected =
        "(0+0j) [Z 15 Y 17 X 19 X 23 X 31 X 63 ] +\n"
        "(1+0j) [Z 15 Z 16 X 17 X 19 X 23 X 31 X 63 ]";

    EXPECT_EQ(res.get_term_count(), 2);
    for (int i = 0; i < res.get_term_count(); i++) {
        auto term = res.get_term(i);
        if (term.second == ex1)
            EXPECT_EQ(term.first, 0i);
        else if (term.second == ex2)
            EXPECT_EQ(term.first, 1.0);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n"
                          << res_expected << "\nres:\n"
                          << res.to_string();
    }
}

TEST(BravyiKitaevTest, bravyi_kitaevTest2) {
    FermionOperator op1;
    op1.add_term(1.0, "50");
    op1.add_term(1.0, "50^");

    Observable res = bravyi_kitaev(op1, 100);

    MultiQubitPauliOperator ex1("Z 31 Z 47 Z 49 Y 50 X 51 X 55 X 63");
    MultiQubitPauliOperator ex2("Z 31 Z 47 Z 49 X 50 X 51 X 55 X 63");

    std::string res_expected =
        "(0+0j) [Z 31 Z 47 Z 49 Y 50 X 51 X 55 X 63 ] +\n"
        "(1+0j) [Z 31 Z 47 Z 49 X 50 X 51 X 55 X 63 ]";

    EXPECT_EQ(res.get_term_count(), 2);
    for (int i = 0; i < res.get_term_count(); i++) {
        auto term = res.get_term(i);
        if (term.second == ex1)
            EXPECT_EQ(term.first, 0i);
        else if (term.second == ex2)
            EXPECT_EQ(term.first, 1.0);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n"
                          << res_expected << "\nres:\n"
                          << res.to_string();
    }
}

TEST(BravyiKitaevTest, bravyi_kitaevTest3) {
    FermionOperator op1;
    op1.add_term(1.0, "73");
    op1.add_term(1.0, "73^");

    Observable res = bravyi_kitaev(op1, 100);

    MultiQubitPauliOperator ex1("Z 63 Z 71 Y 73 X 75 X 79 X 95");
    MultiQubitPauliOperator ex2("Z 63 Z 71 Z 72 X 73 X 75 X 79 X 95");

    std::string res_expected =
        "(0+0j) [Z 63 Z 71 Y 73 X 75 X 79 X 95 ] +\n"
        "(1+0j) [Z 63 Z 71 Z 72 X 73 X 75 X 79 X 95 ]";

    EXPECT_EQ(res.get_term_count(), 2);
    for (int i = 0; i < res.get_term_count(); i++) {
        auto term = res.get_term(i);
        if (term.second == ex1)
            EXPECT_EQ(term.first, 0i);
        else if (term.second == ex2)
            EXPECT_EQ(term.first, 1.0);
        else
            ADD_FAILURE() << "Result IS EXPECTED:\n"
                          << res_expected << "\nres:\n"
                          << res.to_string();
    }
}