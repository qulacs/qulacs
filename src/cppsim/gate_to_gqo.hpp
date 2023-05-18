#include <algorithm>
#include <cassert>
#include <functional>
#include <sstream>

#include "gate.hpp"
#include "general_quantum_operator.hpp"

GeneralQuantumOperator* to_general_quantum_operator(
    const QuantumGateBase* gate, UINT GQO_qubits, double tol = 1e-6);
