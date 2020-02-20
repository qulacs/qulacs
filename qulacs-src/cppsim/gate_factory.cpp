#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include "gate_factory.hpp"
#include "gate.hpp"
#include "gate_named_one.hpp"
#include "gate_named_two.hpp"
#include "gate_named_pauli.hpp"
#include "gate_matrix.hpp"
#include "gate_matrix_sparse.hpp"
#include "gate_reversible.hpp"
#include "gate_reflect.hpp"
#include "type.hpp"
#include <Eigen/QR>

namespace gate{
    ComplexMatrix get_IBMQ_matrix(double theta, double phi, double lambda);

	QuantumGateBase* Identity(UINT qubit_index){
        return new ClsIGate(qubit_index);
    }
    QuantumGateBase* X(UINT qubit_index){
        return new ClsXGate(qubit_index);
    }
    QuantumGateBase* Y(UINT qubit_index){
        return new ClsYGate(qubit_index);
    }
    QuantumGateBase* Z(UINT qubit_index){
        return new ClsZGate(qubit_index);
    }
    QuantumGateBase* H(UINT qubit_index) {
        return new ClsHGate(qubit_index);
    }
    QuantumGateBase* S(UINT qubit_index) {
        return new ClsSGate(qubit_index);
    }
    QuantumGateBase* Sdag(UINT qubit_index) {
        return new ClsSdagGate(qubit_index);
    }
    QuantumGateBase* T(UINT qubit_index) {
        return new ClsTGate(qubit_index);
    }
    QuantumGateBase* Tdag(UINT qubit_index) {
        return new ClsTdagGate(qubit_index);
    }
    QuantumGateBase* sqrtX(UINT qubit_index) {
        return new ClsSqrtXGate(qubit_index);
    }
    QuantumGateBase* sqrtXdag(UINT qubit_index) {
        return new ClsSqrtXdagGate(qubit_index);
    }
    QuantumGateBase* sqrtY(UINT qubit_index) {
        return new ClsSqrtYGate(qubit_index);
    }
    QuantumGateBase* sqrtYdag(UINT qubit_index) {
        return new ClsSqrtYdagGate(qubit_index);
    }
    QuantumGateBase* P0(UINT qubit_index) {
        return new ClsP0Gate(qubit_index);
    }
    QuantumGateBase* P1(UINT qubit_index) {
        return new ClsP1Gate(qubit_index);
    }
    QuantumGateBase* RX(UINT qubit_index,double angle){
        return new ClsRXGate(qubit_index,angle);
    }
    QuantumGateBase* RY(UINT qubit_index, double angle) {
        return new ClsRYGate(qubit_index, angle);
    }
    QuantumGateBase* RZ(UINT qubit_index, double angle) {
        return new ClsRZGate(qubit_index, angle);
    }

    ComplexMatrix get_IBMQ_matrix(double theta, double phi, double lambda) {
        CPPCTYPE im(0, 1);
        CPPCTYPE exp_val1 = exp(im * phi);
        CPPCTYPE exp_val2 = exp(im * lambda);
        CPPCTYPE cos_val = cos(theta / 2);
        CPPCTYPE sin_val = sin(theta / 2);

        ComplexMatrix matrix(2, 2);
        matrix(0,0) = cos_val;
        matrix(0,1) = -exp_val2 * sin_val;
        matrix(1,0) = exp_val1 * sin_val;
        matrix(1,1) = exp_val1 * exp_val2 * cos_val;
        return matrix;
    }
    QuantumGateBase* U1(UINT qubit_index, double lambda) {
        ComplexMatrix matrix = get_IBMQ_matrix(0, 0, lambda);
        std::vector<UINT> vec;
        vec.push_back(qubit_index);
        return new QuantumGateMatrix(vec, matrix);
    }
    QuantumGateBase* U2(UINT qubit_index, double phi, double lambda) {
        ComplexMatrix matrix = get_IBMQ_matrix(M_PI / 2, phi, lambda);
        std::vector<UINT> vec;
        vec.push_back(qubit_index);
        return new QuantumGateMatrix(vec, matrix);
    }
    QuantumGateBase* U3(UINT qubit_index, double theta, double phi, double lambda) {
        ComplexMatrix matrix = get_IBMQ_matrix(theta, phi, lambda);
        std::vector<UINT> vec;
        vec.push_back(qubit_index);
        return new QuantumGateMatrix(vec, matrix);
    }

    QuantumGateBase* CNOT(UINT control_qubit_index, UINT target_qubit_index) {
		if (control_qubit_index == target_qubit_index) {
			std::cerr << "Error: gate::CNOT(UINT, UINT): control_qubit_index and target_qubit_index has the same value." << std::endl;
			return NULL;
		}
		return new ClsCNOTGate(control_qubit_index, target_qubit_index);
    }
    QuantumGateBase* CZ(UINT control_qubit_index, UINT target_qubit_index) {
		if (control_qubit_index == target_qubit_index) {
			std::cerr << "Error: gate::CZ(UINT, UINT): control_qubit_index and target_qubit_index has the same value." << std::endl;
			return NULL;
		}
		return new ClsCZGate(control_qubit_index, target_qubit_index);
    }
    QuantumGateBase* SWAP(UINT qubit_index1, UINT qubit_index2) {
		if (qubit_index1 == qubit_index2) {
			std::cerr << "Error: gate::SWAP(UINT, UINT): two indices have the same value." << std::endl;
			return NULL;
		}
		return new ClsSWAPGate(qubit_index1, qubit_index2);
    }

    QuantumGateBase* Pauli(std::vector<UINT> target, std::vector<UINT> pauli_id) {
		if (!check_is_unique_index_list(target)) {
			std::cerr << "Error: gate::Pauli(std::vector<UINT> target, std::vector<UINT>pauli_id): target list contains duplicated values." << std::endl;
			return NULL;
		}
		auto pauli = new PauliOperator(target, pauli_id);
        return new ClsPauliGate(pauli);
    }
    QuantumGateBase* PauliRotation(std::vector<UINT> target, std::vector<UINT> pauli_id, double angle) {
		if (!check_is_unique_index_list(target)) {
			std::cerr << "Error: gate::PauliRotation(std::vector<UINT> target, std::vector<UINT>pauli_id, double angle): target list contains duplicated values." << std::endl;
			return NULL;
		}
		auto pauli = new PauliOperator(target, pauli_id,angle);
        return new ClsPauliRotationGate(angle,pauli);
    }

    QuantumGateMatrix* DenseMatrix(UINT target_index, ComplexMatrix matrix) {
        std::vector<UINT> target_list(1, target_index);
        return new QuantumGateMatrix(target_list, matrix);
    }
    QuantumGateMatrix* DenseMatrix(std::vector<UINT> target_list, ComplexMatrix matrix) {
		if (!check_is_unique_index_list(target_list)) {
			std::cerr << "Error: gate::DenseMatrix(std::vector<UINT> target_list, ComplexMatrix matrix): target list contains duplicated values." << std::endl;
			return NULL;
		}
		return new QuantumGateMatrix(target_list, matrix);
    }

	QuantumGateBase* SparseMatrix(std::vector<UINT> target_list, SparseComplexMatrix matrix) {
		if (!check_is_unique_index_list(target_list)) {
			std::cerr << "Error: gate::SparseMatrix(std::vector<UINT> target_list, SparseComplexMatrix matrix): target list contains duplicated values." << std::endl;
			return NULL;
		}
		return new QuantumGateSparseMatrix(target_list, matrix);
	}

	QuantumGateMatrix* RandomUnitary(std::vector<UINT> target_list) {
		if (!check_is_unique_index_list(target_list)) {
			std::cerr << "Error: gate::RandomUnitary(std::vector<UINT> target_list): target list contains duplicated values." << std::endl;
			return NULL;
		}
		Random random;
		UINT qubit_count = (UINT)target_list.size();
		ITYPE dim = 1ULL << qubit_count;
		ComplexMatrix matrix(dim, dim);
		for (ITYPE i = 0; i < dim; ++i) {
			for (ITYPE j = 0; j < dim; ++j) {
				matrix(i, j) = (random.normal() + 1.i * random.normal())/sqrt(2.);
			}
		}
		Eigen::HouseholderQR<ComplexMatrix> qr_solver(matrix);
		ComplexMatrix Q = qr_solver.householderQ();
		// actual R matrix is upper-right triangle of matrixQR
		auto R = qr_solver.matrixQR();
		for (ITYPE i = 0; i < dim; ++i) {
			CPPCTYPE phase = R(i, i) / abs(R(i, i));
			for (ITYPE j = 0; j < dim; ++j) {
				Q(j,i) *= phase;
			}
		}
		return new QuantumGateMatrix(target_list, Q);
	}
	QuantumGateBase* ReversibleBoolean(std::vector<UINT> target_qubit_index_list, std::function<ITYPE(ITYPE,ITYPE)> function_ptr) {
		if (!check_is_unique_index_list(target_qubit_index_list)) {
			std::cerr << "Error: gate::ReversibleBoolean(std::vector<UINT> target_qubit_index_list, std::function<ITYPE(ITYPE,ITYPE)> function_ptr): target list contains duplicated values." << std::endl;
			return NULL;
		}
		return new ClsReversibleBooleanGate(target_qubit_index_list, function_ptr);
	}
	QuantumGateBase* StateReflection(const QuantumStateBase* reflection_state) {
		return new ClsStateReflectionGate(reflection_state);
	}


    QuantumGateBase* BitFlipNoise(UINT target_index, double prob) {
		auto gate0 = X(target_index);
		auto gate1 = Identity(target_index);
        auto new_gate = new QuantumGate_Probabilistic({ prob, 1-prob }, { gate0, gate1 });
		delete gate0;
		delete gate1;
		return new_gate;
    }
    QuantumGateBase* DephasingNoise(UINT target_index, double prob) {
		auto gate0 = Z(target_index);
		auto gate1 = Identity(target_index);
		auto new_gate = new QuantumGate_Probabilistic({ prob, 1-prob }, { gate0, gate1 });
		delete gate0;
		delete gate1;
		return new_gate;
    }
    QuantumGateBase* IndependentXZNoise(UINT target_index, double prob) {
		auto gate0 = X(target_index);
		auto gate1 = Z(target_index);
		auto gate2 = Y(target_index);
		auto gate3 = Identity(target_index);
		double p1 = prob * (1 - prob);
		double p2 = prob * prob;
        auto new_gate = new QuantumGate_Probabilistic({ p1, p1, p2, 1-2*p1-p2}, { gate0, gate1, gate2, gate3});
		delete gate0;
		delete gate1;
		delete gate2;
		return new_gate;
	}
    QuantumGateBase* DepolarizingNoise(UINT target_index, double prob) {
		auto gate0 = X(target_index);
		auto gate1 = Z(target_index);
		auto gate2 = Y(target_index);
		auto gate3 = Identity(target_index);
		auto new_gate = new QuantumGate_Probabilistic({ prob / 3,prob / 3,prob / 3, 1-prob }, { gate0, gate1, gate2, gate3 });
		delete gate0;
		delete gate1;
		delete gate2;
		delete gate3;
		return new_gate;
	}
	QuantumGateBase* TwoQubitDepolarizingNoise(UINT target_index1, UINT target_index2, double prob) {
		if (target_index1 == target_index2) {
			std::cerr << "Error: gate::TwoQubitDepolarizingNoise(UINT, UINT, double): target list contains duplicated values." << std::endl;
			return NULL;
		}
		std::vector<QuantumGateBase*> gate_list;
		for (int i = 0; i < 16; ++i) {
			if (i != 0) {
				UINT pauli_qubit1 = i % 4;
				UINT pauli_qubit2 = i / 4;
				auto gate = Pauli({ target_index1, target_index2 }, { pauli_qubit1, pauli_qubit2 });
				gate_list.push_back(gate);
			}
			else {
				gate_list.push_back(Identity(target_index1));
			}
		}
		std::vector<double> probabilities(16, prob/15);
		probabilities[0] = 1 - prob;
		auto new_gate = new QuantumGate_Probabilistic(probabilities, gate_list);
		for (UINT gate_index = 0; gate_index < 15; ++gate_index) {
			delete gate_list[gate_index];
		}
		return new_gate;
	}
	QuantumGateBase* AmplitudeDampingNoise(UINT target_index, double prob) {
		ComplexMatrix damping_matrix_0(2, 2), damping_matrix_1(2,2);
		damping_matrix_0 << 1, 0, 0, sqrt(1 - prob);
		damping_matrix_1 << 0, sqrt(prob), 0, 0;
		auto gate0 = DenseMatrix({ target_index }, damping_matrix_0);
		auto gate1 = DenseMatrix({ target_index }, damping_matrix_1);
		auto new_gate = new QuantumGate_CPTP({ gate0, gate1 });
		delete gate0;
		delete gate1;
		return new_gate;
	}
	QuantumGateBase* Measurement(UINT target_index, UINT classical_register_address) {
		auto gate0 = P0(target_index);
		auto gate1 = P1(target_index);
        auto new_gate = new QuantumGate_Instrument({ gate0, gate1},classical_register_address);
		delete gate0;
		delete gate1;
		return new_gate;
    }

    QuantumGateBase* create_quantum_gate_from_string(std::string gate_string){
        const char* gateString = gate_string.c_str();
        char* sbuf;
        //ITYPE elementCount;
        std::vector<CPPCTYPE> element;
        const char delim[] = " ";
        std::vector<UINT> targets;
        QuantumGateBase* gate = NULL;
        char* buf = (char*)calloc(strlen(gateString)+1, sizeof(char));
        strcpy(buf, gateString);
        sbuf = strtok(buf , delim);

        if(strcasecmp(sbuf,"I")==0)         gate = gate::Identity( atoi( strtok(NULL , delim) ) );
        else if(strcasecmp(sbuf,"X")==0)    gate = gate::X( atoi( strtok(NULL , delim) ) );
        else if(strcasecmp(sbuf,"Y")==0)    gate = gate::Y( atoi( strtok(NULL , delim) ) );
        else if(strcasecmp(sbuf,"Z")==0)    gate = gate::Z( atoi( strtok(NULL , delim) ) );
        else if(strcasecmp(sbuf,"H")==0)    gate = gate::H( atoi( strtok(NULL , delim) ) );
        else if(strcasecmp(sbuf,"S")==0)    gate = gate::S( atoi( strtok(NULL , delim) ) );
        else if(strcasecmp(sbuf,"Sdag")==0) gate = gate::Sdag( atoi( strtok(NULL , delim) ) );
        else if(strcasecmp(sbuf,"T")==0)    gate = gate::T( atoi( strtok(NULL , delim) ) );
        else if(strcasecmp(sbuf,"Tdag")==0) gate = gate::Tdag( atoi( strtok(NULL , delim) ) );
        else if(strcasecmp(sbuf,"CNOT")==0 || strcasecmp(sbuf,"CX")==0) {
            unsigned int control = atoi( strtok(NULL , delim) );
            unsigned int target = atoi( strtok(NULL , delim) );
            gate = gate::CNOT( control, target );
        }
        else if(strcasecmp(sbuf,"CZ")==0)   {
            unsigned int control = atoi( strtok(NULL , delim) );
            unsigned int target = atoi( strtok(NULL , delim) );
            gate = gate::CZ( control, target );
        }
        else if(strcasecmp(sbuf,"SWAP")==0) {
            unsigned int target1 = atoi( strtok(NULL , delim) );
            unsigned int target2 = atoi( strtok(NULL , delim) );
            gate = gate::SWAP( target1, target2 );
        }
        else if(strcasecmp(sbuf,"U1")==0)   {
            unsigned int target = atoi( strtok(NULL , delim) );
            double theta1 = atof( strtok(NULL , delim) );
            gate = gate::U1( target, theta1 );
        }
        else if(strcasecmp(sbuf,"U2")==0)   {
            unsigned int target = atoi( strtok(NULL , delim) );
            double theta1 = atof( strtok(NULL , delim) );
            double theta2 = atof( strtok(NULL , delim) );
            gate = gate::U2( target, theta1, theta2 );
        }
        else if(strcasecmp(sbuf,"U3")==0)   {
            unsigned int target = atoi( strtok(NULL , delim) );
            double theta1 = atof( strtok(NULL , delim) );
            double theta2 = atof( strtok(NULL , delim) );
            double theta3 = atof( strtok(NULL , delim) );
            gate = gate::U3( target, theta1, theta2, theta3 );
        }
        else if(strcasecmp(sbuf,"RX")==0){
            unsigned int target = atoi( strtok(NULL , delim) );
            double theta = atof( strtok(NULL , delim) );
            gate = gate::RX( target, theta );
        }
        else if(strcasecmp(sbuf,"RY")==0){
            unsigned int target = atoi( strtok(NULL , delim) );
            double theta = atof( strtok(NULL , delim) );
            gate = gate::RY( target, theta );
        }
        else if(strcasecmp(sbuf,"RZ")==0){
            unsigned int target = atoi( strtok(NULL , delim) );
            double theta = atof( strtok(NULL , delim) );
            gate = gate::RZ( target, theta );
        }
        else if(strcasecmp(sbuf,"RM")==0){
            char* pauliStr = strtok(NULL,delim);
            unsigned int targetCount = (UINT)strlen(pauliStr);

            std::vector<UINT> pauli(targetCount, 0);
            for(unsigned int i=0;i<targetCount;i++){
                if(pauliStr[i] == 'x' || pauliStr[i] == 'X') pauli[i] = 1;
                else if(pauliStr[i] == 'y' || pauliStr[i] == 'Y') pauli[i] = 2;
                else if(pauliStr[i] == 'z' || pauliStr[i] == 'Z') pauli[i] = 3;
            }

            targets = std::vector<UINT>(targetCount,0);
            for(unsigned int i=0;i<targetCount;i++){
                targets[i] = atoi( strtok(NULL , delim) );
            }

            double theta = atof(strtok(NULL, delim));
            gate = gate::PauliRotation(targets, pauli, theta);
        }
        else if(strcasecmp(sbuf,"U")==0)  {
            unsigned int targetCount = atoi(strtok(NULL,delim));

            targets = std::vector<UINT>(targetCount,0);
            for(unsigned int i=0;i<targetCount;i++){
                targets[i] = atoi( strtok(NULL , delim) );
            }
            ITYPE dim = 1ULL << targetCount;
            ComplexMatrix matrix(dim, dim);

            for(ITYPE i=0;i<dim*dim;i++){
                char* token;
                token = strtok(NULL, delim);
                matrix(i/dim, i%dim) = atof( token );
                token = strtok(NULL, delim);
                matrix(i/dim, i%dim) += CPPCTYPE(0,1) * atof( token );
            }
            gate = gate::DenseMatrix(targets, matrix);
        }
        free(buf);
        return gate;
    }

}


