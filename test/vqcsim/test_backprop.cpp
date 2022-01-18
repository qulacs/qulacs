#include <gtest/gtest.h>

#include <cppsim/circuit.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>
#include <iostream>
#include <vqcsim/GradCalculator.hpp>
#include <vqcsim/parametric_circuit.hpp>
using namespace std;
TEST(Backprop, BackpropCircuit) {
    ParametricQuantumCircuit kairo(3);
    kairo.add_parametric_RX_gate(0, 2.2);
    kairo.add_parametric_RY_gate(1, 0);
    kairo.add_gate(gate::CNOT(0, 2));
    kairo.add_parametric_RZ_gate(2, 1.4);
    kairo.add_gate(gate::H(1));
    kairo.add_parametric_RY_gate(0, 1);
    kairo.add_gate(gate::CNOT(1, 0));
    kairo.add_gate(gate::H(1));
    kairo.add_parametric_RZ_gate(1, -1);
    kairo.add_gate(gate::H(0));
    kairo.add_gate(gate::CNOT(2, 0));
    kairo.add_parametric_RX_gate(2, 1);
    kairo.add_gate(gate::CNOT(1, 0));
    kairo.add_parametric_RZ_gate(0, 1);
    kairo.add_gate(gate::CNOT(0, 1));
    kairo.add_gate(gate::H(1));
    kairo.add_parametric_RX_gate(1, -1);
    kairo.add_parametric_RY_gate(2, 1);
    //回路に適当にゲートを加える
    // cout<<kairo<<endl;
    QuantumState state(3);
    state.set_zero_state();
    kairo.update_quantum_state(&state);
    // cout<<state<<endl;
    Observable observable(3);
    observable.add_operator(1, "X 0");
    observable.add_operator(1.2, "Y 1");
    observable.add_operator(1.5, "Z 2");
    // cout<<observable.get_expectation_value(&state)<<endl<<endl;
    vector<double> kaku = {2.2, 0, 1.4, 1, -1, 1, 1, -1, 1};
    // vector<double> kaku={2.2,1.4};
    GradCalculator wrakln;
    auto bibun = wrakln.calculate_grad(kairo, observable, kaku);
    // for(auto it:bibun){cout<<it<<endl;}
    // cout<<"de"<<endl;
    // culculate_gradした後は、パラメータがぐちゃぐちゃになるので、再セット
    // a
    kairo.set_parameter(0, 2.2);
    kairo.set_parameter(1, 0);
    kairo.set_parameter(2, 1.4);
    kairo.set_parameter(3, 1);
    kairo.set_parameter(4, -1);
    kairo.set_parameter(5, 1);
    kairo.set_parameter(6, 1);
    kairo.set_parameter(7, -1);
    kairo.set_parameter(8, 1);
    auto bk = kairo.backprop(&observable);
    for (int i = 0; i < 9; i++) {
        cerr << bk[i] << " " << bibun[i].real() << endl;
        ASSERT_NEAR(bk[i], bibun[i].real(), 1e-10);
    }
    // for(auto it:bk){cout<<it<<endl;}
    // cout<<"de"<<endl;
    // for(auto it:bibun){cout<<it<<endl;}
    // cout<<"de"<<endl;
    // cout<<observable.get_expectation_value(&state)<<endl<<endl;
    // cerr<<(*gate::H(1))<<endl;
}
