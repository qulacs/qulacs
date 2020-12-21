#include <gtest/gtest.h>
#include "../util/util.h"

#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/pauli_operator.hpp>

TEST(PauliOperatorTest,BasicTest) {
	int n = 10;
	double coef = 2.0;
	std::string Pauli_string = "X 0 Y 1 Z 2 I 3 X 4 Y 6 Z 5 I 8 X 7 Y 9";
	PauliOperator pauli = PauliOperator(Pauli_string, coef);
	QuantumState state = QuantumState(n);
	state.set_Haar_random_state();
	CPPCTYPE value = pauli.get_expectation_value(&state);
	ASSERT_NE(value, CPPCTYPE(0,0));
}

TEST(PauliOperatorTest, PauliQubitOverflow) {
	//This test is based on issue #259. Thanks tsuvihatu!
	int n = 2;
	double coef = 2.0;
	std::string Pauli_string = "X 0 X 1 X 3";
	PauliOperator pauli = PauliOperator(Pauli_string, coef);
	QuantumState state = QuantumState(n);
	state.set_Haar_random_state();
	CPPCTYPE value = pauli.get_expectation_value(&state);
	ASSERT_NE(value, value); // (value != value is true) if and only if value is NaN.
}

TEST(PauliOperatorTest, BrokenPauliString) {
	int n = 5;
	double coef = 2.0;
	std::string Pauli_string = "X 0 X Z 1 Y 2";
	PauliOperator pauli = PauliOperator(Pauli_string, coef);
	int PauliSize = pauli.get_index_list().size();
	ASSERT_EQ(PauliSize,2);
}

TEST(PauliOperatorTest, SpacedPauliString) {
	//This test is based on issue #257.  Thanks r-imai-quantum!
	int n = 5;
	double coef = 2.0;
	std::string Pauli_string = "X 0 Y 1 ";
	PauliOperator pauli = PauliOperator(Pauli_string, coef);
	int PauliSize = pauli.get_index_list().size();
	ASSERT_EQ(PauliSize, 2);
}

TEST(PauliOperatorTest, PartedPauliString) {
	int n = 5;
	double coef = 2.0;
	std::string Pauli_string = "X 0 Y ";
	PauliOperator pauli = PauliOperator(Pauli_string, coef);
	int PauliSize = pauli.get_index_list().size();
	ASSERT_EQ(PauliSize, 1);
}