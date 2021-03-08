#include <gtest/gtest.h>

#include <cereal/archives/portable_binary.hpp>
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
        std::ofstream os("out4.cereal", std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(os);
        archive(*hoge);
        hoge->update_quantum_state(&a);
        os.close();
    }
    {
        QuantumGateBasic* hoge = gate::X(2);
        std::ifstream is("out4.cereal", std::ios::binary);
        cereal::PortableBinaryInputArchive archive(is);
        archive(*hoge);
        hoge->update_quantum_state(&b);
    }
    // StateVector applied by QuantumGate should be same.
    for (int i = 0; i < (1 << 6); ++i) {
        ASSERT_NEAR(abs(a.data_cpp()[i] - b.data_cpp()[i]), 0, 1e-7);
    }
}

TEST(CerealTest, Serialize_SparseComplexMatrix) {
    // Just Check whether they run without Runtime Errors.
    StateVector a(6), b(6);
    a.set_zero_state();
    b.set_zero_state();
    {
        SparseComplexMatrix mat((1 << 5), (1 << 5));
        std::vector<Eigen::Triplet<CPPCTYPE>> TripletList;
        for (int i = 0; i < (1 << 5); ++i) {
            TripletList.push_back(
                Eigen::Triplet<CPPCTYPE>((1 << 5) - i - 1, i, i + 1));
            TripletList.push_back(Eigen::Triplet<CPPCTYPE>(i, i, i + 1));
        }
        mat.setFromTriplets(TripletList.begin(), TripletList.end());

        QuantumGateBasic* hoge =
            QuantumGateBasic::SparseMatrixGate({0, 1, 2, 3, 4}, mat, {});
        std::ofstream os("out1.cereal", std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(os);
        archive(*hoge);
        hoge->update_quantum_state(&a);
        os.close();
    }
    {
        QuantumGateBasic* hoge = gate::X(2);
        std::ifstream is("out1.cereal", std::ios::binary);
        cereal::PortableBinaryInputArchive archive(is);
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

    std::unique_ptr<QuantumGateBase> gate_A, gate_B;
    {
        gate_A.reset(gate::TwoQubitDepolarizingNoise(0, 1, 0.5));
        std::ofstream os("out2.cereal", std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(os);
        archive(gate_A);
        gate_A->update_quantum_state(&a);
        os.close();
    }
    {
        std::ifstream is("out2.cereal", std::ios::binary);
        cereal::PortableBinaryInputArchive archive(is);
        archive(gate_B);
        gate_B->update_quantum_state(&b);
    }
    // gate_A should be equal to gate_B.
    // we'll check this.
    ASSERT_EQ(gate_A->get_cumulative_distribution(),
        gate_B->get_cumulative_distribution());
    std::vector<QuantumGateBase*> A_kraus_list, B_kraus_list;
    A_kraus_list = gate_A->get_kraus_list();
    B_kraus_list = gate_B->get_kraus_list();
    ASSERT_EQ(A_kraus_list.size(), B_kraus_list.size());
    for (UINT i = 0; i < A_kraus_list.size(); ++i) {
        a.set_zero_state();
        b.set_zero_state();
        A_kraus_list[i]->update_quantum_state(&a);
        B_kraus_list[i]->update_quantum_state(&b);
        for (int i = 0; i < (1 << 6); ++i) {
            ASSERT_NEAR(abs(a.data_cpp()[i] - b.data_cpp()[i]), 0, 1e-7);
        }
    }
}

TEST(CerealTest, serealize_QuantumCircuit) {
    // Just Check whether they run without Runtime Errors.
    StateVectorCpu a(10), b(10);
    a.set_zero_state();
    b.set_zero_state();
    QuantumCircuit inputs(10), outputs(1);
    inputs.add_gate(gate::X(0));
    inputs.add_gate(gate::H(1));
    inputs.update_quantum_state(&a);
    {
        // serialize QuantumCircuit
        std::ofstream os("out3.cereal", std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(os);
        archive(inputs);
    }
    {
        // deserialize QuantumCircuit
        std::ifstream is("out3.cereal", std::ios::binary);
        cereal::PortableBinaryInputArchive archive(is);
        archive(outputs);
    }
    outputs.update_quantum_state(&b);

    // StateVector applied by QuantumCircuit should be same.
    for (int i = 0; i < (1 << 10); ++i) {
        ASSERT_NEAR(abs(a.data_cpp()[i] - b.data_cpp()[i]), 0, 1e-7);
    }
}

TEST(CerealTest, serealize_quantum_gate_basic_for_python) {
    // Just Check whether they run without Runtime Errors.
    QuantumGateBase* gate1 = QuantumGateBasic::DenseMatrixGate(
        {0, 1}, Eigen::MatrixXcd::Identity(4, 4));
    QuantumGateBase* gate2 =
        QuantumGateBasic::DenseMatrixGate({2}, Eigen::MatrixXcd::Zero(2, 2));
    std::string ss = gate1->dump_as_byte();
    gate2->load_from_byte(ss);

    // Check equivalence
    StateVectorCpu a(10), b(10);
    a.set_Haar_random_state();
    b.load(&a);
    gate1->update_quantum_state(&a);
    gate2->update_quantum_state(&b);
    for (int i = 0; i < (1 << 10); ++i) {
        ASSERT_NEAR(abs(a.data_cpp()[i] - b.data_cpp()[i]), 0, 1e-7);
    }
}

TEST(CerealTest, serealize_QuantumCircuit_for_python) {
    // Just Check whether they run without Runtime Errors.
    QuantumCircuit circuit1(10), circuit2(1);
    circuit1.add_gate(gate::X(0));
    circuit1.add_gate(gate::H(1));

    std::string ss = circuit1.dump_as_byte();
    circuit2.load_from_byte(ss);

    // Check equivalence
    StateVectorCpu a(10), b(10);
    a.set_Haar_random_state();
    b.load(&a);
    circuit1.update_quantum_state(&a);
    circuit2.update_quantum_state(&b);
    for (int i = 0; i < (1 << 10); ++i) {
        ASSERT_NEAR(abs(a.data_cpp()[i] - b.data_cpp()[i]), 0, 1e-7);
    }
}
