#include <gtest/gtest.h>

#include <cppsim_experimental/circuit.hpp>
#include <cppsim_experimental/gate.hpp>
#include <cppsim_experimental/noisesimulator.hpp>
#include <cppsim_experimental/state.hpp>

#include <cereal/archives/binary.hpp>
#include <iostream>
#include <fstream>

#include "../util/util.h"

TEST(CerealTest, Random_with_State_Test) {
    // Just Check whether they run without Runtime Errors.
    {
        ComplexMatrix mat(8, 8);

            mat << 1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0;
        QuantumGateBasic* hoge = QuantumGateBasic::DenseMatrixGate(
        {0,1,2}, mat, {});
        std::ofstream os("out.cereal", std::ios::binary);
        cereal::BinaryOutputArchive archive(os);

        archive(*hoge);
        std::cout << *hoge << std::endl;
    }
    std::cout << "=====================================================\n";
    {
        QuantumGateBasic* hoge = gate::X(0);
        std::ifstream is("out.cereal", std::ios::binary);
        cereal::BinaryInputArchive archive(is);
        archive(*hoge);
        std::cout << *hoge << std::endl;
    }
}