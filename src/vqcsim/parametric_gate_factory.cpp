#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cppsim/utility.hpp>
#include <cppsim/gate_factory.hpp>
#include "parametric_gate_factory.hpp"
#include "parametric_gate.hpp"

namespace gate {
    QuantumGate_SingleParameter* ParametricRX(UINT target_qubit_index, double initial_angle) {
        return new ClsParametricRXGate(target_qubit_index, initial_angle);
    }
    QuantumGate_SingleParameter* ParametricRY(UINT target_qubit_index, double initial_angle) {
        return new ClsParametricRYGate(target_qubit_index, initial_angle);
    }
    QuantumGate_SingleParameter* ParametricRZ(UINT target_qubit_index, double initial_angle) {
        return new ClsParametricRZGate(target_qubit_index, initial_angle);
    }
    QuantumGate_SingleParameter* ParametricPauliRotation(std::vector<UINT> target, std::vector<UINT> pauli_id, double initial_angle) {
		if (!check_is_unique_index_list(target)) {
			std::cerr << "Error: gate::ParametricPauliRotation(std::vector<UINT>, std::vector<UINT>, double): target qubit list contains duplicated values." << std::endl;
			return NULL;
		}
		auto pauli = new PauliOperator(target, pauli_id, initial_angle);
        return new ClsParametricPauliRotationGate(initial_angle, pauli);
    }


    QuantumGateBase* create_parametric_quantum_gate_from_string(std::string gate_string) {

        auto non_parametric_gate = gate::create_quantum_gate_from_string(gate_string);
        if (non_parametric_gate != NULL) return non_parametric_gate;

        const char* gateString = gate_string.c_str();
        char* sbuf;
        //ITYPE elementCount;
        std::vector<CPPCTYPE> element;
        const char delim[] = " ";
        std::vector<UINT> targets;
        QuantumGateBase* gate = NULL;
        char* buf = (char*)calloc(strlen(gateString) + 1, sizeof(char));
        strcpy(buf, gateString);
        sbuf = strtok(buf, delim);

        if(strcasecmp(sbuf,"PRX")==0){
            unsigned int target = atoi( strtok(NULL , delim) );
            gate = gate::ParametricRX(target);
        }
        else if(strcasecmp(sbuf,"PRY")==0){
            unsigned int target = atoi( strtok(NULL , delim) );
            gate = gate::ParametricRY(target);
        }
        else if(strcasecmp(sbuf,"PRZ")==0){
            unsigned int target = atoi( strtok(NULL , delim) );
            gate = gate::ParametricRZ(target);
        }
        else if (strcasecmp(sbuf, "PPR") == 0) {
            char* pauliStr = strtok(NULL, delim);
            unsigned int targetCount = (UINT)strlen(pauliStr);

            std::vector<UINT> pauli(targetCount, 0);
            for (unsigned int i = 0; i < targetCount; i++) {
                if (pauliStr[i] == 'x' || pauliStr[i] == 'X') pauli[i] = 1;
                else if (pauliStr[i] == 'y' || pauliStr[i] == 'Y') pauli[i] = 2;
                else if (pauliStr[i] == 'z' || pauliStr[i] == 'Z') pauli[i] = 3;
            }

            targets = std::vector<UINT>(targetCount, 0);
            for (unsigned int i = 0; i < targetCount; i++) {
                targets[i] = atoi(strtok(NULL, delim));
            }
            gate = gate::ParametricPauliRotation(targets, pauli);
        }
        free(buf);
        return gate;
    }

}


