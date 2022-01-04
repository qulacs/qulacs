
#include "gate_wrapper.hpp"

#include "gate_basic.hpp"

namespace gate {
DllExport QuantumGateWrapped* DepolarizingNoise(UINT index, double prob) {
    auto ptr = QuantumGateWrapped::ProbabilisticGate(
        {gate::Identity(index), gate::X(index), gate::Y(index), gate::Z(index)},
        {1 - prob, prob / 3, prob / 3, prob / 3}, "", true);
    return ptr;
}
DllExport QuantumGateWrapped* IndependentXZNoise(UINT index, double prob) {
    auto ptr = QuantumGateWrapped::ProbabilisticGate(
        {gate::Identity(index), gate::X(index), gate::Z(index), gate::Y(index)},
        {(1 - prob) * (1 - prob), prob * (1 - prob), (1 - prob) * prob,
            prob * prob},
        "", true);
    return ptr;
}

DllExport QuantumGateWrapped* TwoQubitDepolarizingNoise(
    UINT index1, UINT index2, double prob) {
    std::vector<QuantumGateBase*> gates;
    std::vector<double> probs;
    probs.push_back(1 - prob);
    gates.push_back(gate::Identity(index1));
    for (UINT i = 1; i < 16; ++i) {
        ComplexMatrix matrix;
        get_Pauli_matrix(matrix, {i % 4, i / 4});
        auto gate = QuantumGateBasic::DenseMatrixGate({index1, index2}, matrix);
        //auto gate = QuantumGateBasic::PauliMatrixGate(
        //    {index1, index2}, {i % 4, i / 4}, 0.);
        gates.push_back(gate);
        probs.push_back(prob / 15);
    }
    auto ptr = QuantumGateWrapped::ProbabilisticGate(gates, probs, "", true);
    return ptr;
}
DllExport QuantumGateWrapped* BitFlipNoise(UINT index, double prob) {
    auto ptr = QuantumGateWrapped::ProbabilisticGate(
        {gate::Identity(index), gate::X(index)}, {1 - prob, prob}, "", true);
    return ptr;
}
DllExport QuantumGateWrapped* DephasingNoise(UINT index, double prob) {
    auto ptr = QuantumGateWrapped::ProbabilisticGate(
        {gate::Identity(index), gate::Z(index)}, {1 - prob, prob}, "", true);
    return ptr;
}
DllExport QuantumGateWrapped* AmplitudeDampingNoise(UINT index, double prob) {
    ComplexMatrix damping_matrix_0(2, 2), damping_matrix_1(2, 2);
    damping_matrix_0 << 1, 0, 0, sqrt(1 - prob);
    damping_matrix_1 << 0, sqrt(prob), 0, 0;
    auto gate0 = QuantumGateBasic::DenseMatrixGate({index}, damping_matrix_0);
    auto gate1 = QuantumGateBasic::DenseMatrixGate({index}, damping_matrix_1);
    auto ptr = QuantumGateWrapped::CPTP({gate0, gate1});
    return ptr;
}
DllExport QuantumGateWrapped* Measurement(UINT index, std::string name) {
    auto ptr = QuantumGateWrapped::Instrument(
        {gate::P0(index), gate::P1(index)}, name, true);
    return ptr;
}

}  // namespace gate
