
#include <unordered_map>
#include <string>
#include <nlohmann/json.hpp>
#include <cppsim/type.hpp>
#include <vector>
#include <bitset>


enum QulacsGates {
    // ClsOneQubitGate
    // IDENTITY,
    // X,
    // Y,
    // Z,
    // H,
    S,
    SDAG,
    T,
    TDAG,
    SQRTX, // SX????
    SQRTXDAG,
    SQRTY,
    SQRTYDAG,
    P0,
    P1,
    // QuantumGateMatrix
    U1,
    U2,
    U3,
    // ClsOneQubitRotationGate
    // RX,
    // RY,
    // RZ,
    ROTINVX,
    ROTINVY,
    ROTINVZ,
    ROTX, // RX???
    ROTY, // RY???
    ROTZ, // RZ???
    // ClsOneControlOneTargetGate
    CNOT, // CX???
    // CZ,
    // ClsTwoQubitGate
    // SWAP,
    // ECR,
    // ClsNpairQubitGate
    FUSEDSWAP,
    // ClsPauliGate
    PAULI,
    // ClsPauliRotationGate
    PAULIROTATION,
    // QuantumGateMatrix
    DENSEMATRIX, // 2 implementacións
    // QuantumGateSparseMatrix
    SPARSEMATRIX,
    // QuantumGateDiagonalMatrix
    DIAGONALMATRIX,
    // QuantumGateMatrix
    RANDOMUNITARY, // 2 implementacións
    // ClsReversibleBooleanGate
    REVERSIBLEBOOLEAN,
    // ClsStateReflectionGate
    STATEREFLECTION,
    // QuantumGate_LinearCombination
    LINEARCOMBINATION,
    // QuantumGate_Probabilistic
    BITFLIPNOISE, // 2 implementacións
    DEPHASINGNOISE, // 2 implementacións
    INDEPENDENTXZNOISE, // 2 implementacións
    DEPOLARIZINGNOISE, // 2 implementacións
    TWOQUBITDEPOLARIZINGNOISE, // 2 implementacións
    // QuantumGate_CPTP
    AMPLITUDEDAMPINGNOISE, // 2 implementacións
    // QuantumGate_Instrument
    MEASUREMENT, // 2 implementacións
    MULTIQUBITPAULIMEASUREMENT, // 2 implementacións 
    // ClsNoisyEvolution
    NOISYEVOLUTION, 
    // ClsNoisyEvolution_fast
    NOISYEVOLUTION_FAST,
    // ClsNoisyEvolution_auto
    NOISYEVOLUTION_AUTO,
 

    MEASURE,
};

const std::unordered_map<std::string, int> INSTRUCTIONS_MAP = {
    {"s", S},
    {"sdag", SDAG},
    {"t", T},
    {"tdag", TDAG},
    {"sqrtx", SQRTX},
    {"sqrtxdag", SQRTXDAG},
    {"sqrty", SQRTY},
    {"sqrtydag", SQRTYDAG},
    {"p0", P0},
    {"p1", P1},
    {"u1", U1},
    {"u2", U2},
    {"u3", U3},
    {"rotinvx", ROTINVX},
    {"rotinvy", ROTINVY},
    {"rotinvz", ROTINVZ},
    {"rotx", ROTX},
    {"roty", ROTY},
    {"rotz", ROTZ},
    {"cnot", CNOT},
    {"fusedswap", FUSEDSWAP},
    {"pauli", PAULI},
    {"paulirotation", PAULIROTATION},
    {"densematrix", DENSEMATRIX},
    {"sparsematrix", SPARSEMATRIX},
    {"diagonalmatrix", DIAGONALMATRIX},
    {"randomunitary", RANDOMUNITARY},
    {"reversibleboolean", REVERSIBLEBOOLEAN},
    {"statereflection", STATEREFLECTION},
    {"linearcombination", LINEARCOMBINATION},
    {"bitflipnoise", BITFLIPNOISE},
    {"dephasingnoise", DEPHASINGNOISE},
    {"independentxznoise", INDEPENDENTXZNOISE},
    {"depolarizingnoise", DEPOLARIZINGNOISE}, 
    {"twoqubitdepolarizingnoise", TWOQUBITDEPOLARIZINGNOISE},
    {"amplitudedampingnoise", AMPLITUDEDAMPINGNOISE},
    {"measurement", MEASUREMENT},
    {"multiqubitpaulimeasurement", MULTIQUBITPAULIMEASUREMENT},
    {"noisyevolution", NOISYEVOLUTION},
    {"noisyevolution_fast", NOISYEVOLUTION_FAST},
    {"noisyevolution_auto", NOISYEVOLUTION_AUTO},


    {"measure", MEASURE},
};






using json = nlohmann::json;

inline json convert_to_counts(const std::vector<ITYPE>& result, int num_qubits)
{
    std::unordered_map<std::string, int> counts;

    for (ITYPE value : result) {
        // convertir value a bitset y sacar su string con padding
        std::bitset<64> bs(value);
        std::string bitstring = bs.to_string();

        if (num_qubits <= 0) {
            // evitar substr con valor inválido
            bitstring = "";
        } else if (num_qubits < 64) {
            bitstring = bitstring.substr(64 - num_qubits);
        } // else num_qubits >= 64 -> usamos todo el std::bitset<64>

        counts[bitstring]++;
    }

    json j;
    for (const auto& kv : counts) {
        j[kv.first] = kv.second;
    }

    return j;
}
