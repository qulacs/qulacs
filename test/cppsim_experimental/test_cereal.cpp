#include <gtest/gtest.h>

#include <cereal/archives/binary.hpp>
#include <cppsim_experimental/circuit.hpp>
#include <cppsim_experimental/gate.hpp>
#include <cppsim_experimental/noisesimulator.hpp>
#include <cppsim_experimental/state.hpp>
#include <fstream>
#include <iostream>

#include "../util/util.h"

TEST(CerealTest, Serialize_ComplexMatrix) {
    // Just Check whether they run without Runtime Errors.
    StateVector a(6), b(6);
    a.set_zero_state();
    b.set_zero_state();
    {
        ComplexMatrix mat(8, 8);

        mat << 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0;
        QuantumGateBasic* hoge =
            QuantumGateBasic::DenseMatrixGate({0, 1, 2}, mat, {});
        std::ofstream os("out1.cereal", std::ios::binary);
        cereal::BinaryOutputArchive archive(os);
        archive(*hoge);
        hoge->update_quantum_state(&a);
        os.close();
    }
    sleep(3);
    {
        QuantumGateBasic* hoge = gate::X(2);
        std::ifstream is("out1.cereal", std::ios::binary);
        cereal::BinaryInputArchive archive(is);
        archive(*hoge);
        hoge->update_quantum_state(&b);
    }
    // StateVector applied by QuantumGate should be same.
    for (int i = 0; i < (1 << 6); ++i) {
        ASSERT_NEAR(abs(a.data_cpp()[i] - b.data_cpp()[i]), 0, 1e-7);
    }
}

TEST(CerealTest, Serialize_QuantumGateWrapped) {
    // Just Check whether they run without Runtime Errors.
    StateVector a(6), b(6);
    a.set_zero_state();
    b.set_zero_state();
    {
        QuantumGateBase* tmp = gate::TwoQubitDepolarizingNoise(0, 1, 0.5);
        std::unique_ptr<QuantumGateBase> hoge(tmp);
        std::ofstream os("out2.cereal", std::ios::binary);
        cereal::BinaryOutputArchive archive(os);
        archive(hoge);
        hoge->update_quantum_state(&a);
        hoge.release();
        delete tmp;
        os.close();
    }
    sleep(3);
    {
        QuantumGateBase* tmp = gate::X(0);
        std::unique_ptr<QuantumGateBase> hoge(tmp);
        std::ifstream is("out2.cereal", std::ios::binary);
        cereal::BinaryInputArchive archive(is);
        archive(hoge);
        //hoge->update_quantum_state(&b);
    }
}

TEST(CerealTest, serealize_QuantumCircuit) {
    // Just Check whether they run without Runtime Errors.
    StateVectorCpu a(10), b(10);
    a.set_zero_state();
    b.set_zero_state();
    QuantumCircuit inputs(10), outputs(1);
    {
        // serialize QuantumCircuit
        inputs.add_gate(gate::X(0));
        inputs.add_gate(gate::H(1));
        inputs.update_quantum_state(&a);
        std::ofstream os("out3.cereal", std::ios::binary);
        cereal::BinaryOutputArchive archive(os);
        archive(inputs);
        os.close();
    }
    sleep(3);
    {
        // deserialize QuantumCircuit
        std::ifstream is("out3.cereal", std::ios::binary);
        cereal::BinaryInputArchive archive(is);
        archive(outputs);
        outputs.update_quantum_state(&b);
    }

    // StateVector applied by QuantumCircuit should be same.
    for (int i = 0; i < (1 << 10); ++i) {
        ASSERT_NEAR(abs(a.data_cpp()[i] - b.data_cpp()[i]), 0, 1e-7);
    }
}
