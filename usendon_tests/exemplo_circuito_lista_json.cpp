#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <nlohmann/json.hpp>

#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/gate_factory.hpp>

using json = nlohmann::json;


/* struct GateOp {
    std::string name;
    std::vector<int> qubits;
    std::vector<double> params;  
}; */

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

int main() {

    std::vector<json> ops = {
        {
            {"name", "h"},
            {"qubits", {0}}
        },
        {
            {"name", "cnot"},
            {"qubits", {0, 1}}
        }
    };

    int nqubits = 3;

    QuantumState state(nqubits);
    QuantumCircuit circuit(nqubits);

    for (const auto& op : ops) {

        if (op.at("name")== "h") circuit.add_H_gate(op.at("qubits")[0]);
        else if (op.at("name") == "x") circuit.add_X_gate(op.at("qubits")[0]);
        else if (op.at("name") == "y") circuit.add_Y_gate(op.at("qubits")[0]);
        else if (op.at("name") == "z") circuit.add_Z_gate(op.at("qubits")[0]);
        else if (op.at("name") == "cnot") circuit.add_CNOT_gate(op.at("qubits")[0], op.at("qubits")[1]);
        else if (op.at("name") == "ecr") circuit.add_ECR_gate(op.at("qubits")[0], op.at("qubits")[1]);
        else if (op.at("name") == "cz") circuit.add_CZ_gate(op.at("qubits")[0], op.at("qubits")[1]);
        else if (op.at("name") == "swap") circuit.add_SWAP_gate(op.at("qubits")[0], op.at("qubits")[1]);
        else if (op.at("name") == "rx") circuit.add_RX_gate(op.at("qubits")[0], op.at("params")[0]);
        else if (op.at("name") == "ry") circuit.add_RY_gate(op.at("qubits")[0], op.at("params")[0]);
        else if (op.at("name") == "rz") circuit.add_RZ_gate(op.at("qubits")[0], op.at("params")[0]);
        else
            std::cerr << "Puerta no reconocida: " << op.at("name") << std::endl;
    }

    circuit.update_quantum_state(&state);
    int shots = 10;
    std::vector<ITYPE> result = state.sampling(shots);

    std::cout << "Result:" << std::endl;
    for (ITYPE& count : result) {
        std::cout << count << std::endl;
    }

    return 0;
}
