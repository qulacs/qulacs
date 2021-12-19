
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
        auto gate = QuantumGateBasic::PauliMatrixGate(
            {index1, index2}, {i % 4, i / 4}, 0.);
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
}  // namespace gate
