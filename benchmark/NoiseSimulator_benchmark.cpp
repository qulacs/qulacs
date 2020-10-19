#define _USE_MATH_DEFINES 

#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/noisesimulator.hpp>

void random_1Q_gate(int qubitID,QuantumCircuit *val){
	int prob = rand() % 3;
	if(prob == 0){
		val -> add_sqrtX_gate(qubitID);
	}else if(prob == 1){
		val -> add_sqrtY_gate(qubitID);
	}else{
		ComplexMatrix matrix(2,2);
		matrix << 1.0/sqrt(2.0),-sqrt(1.0i/2.0),sqrt(-1.0i/2.0),1.0/sqrt(2.0);
		QuantumGateMatrix* tmp = gate::DenseMatrix(qubitID,matrix);
		val -> add_gate(tmp);
	}
}

void grid_2Q_gate(std::vector<UINT> inputs,QuantumCircuit *ans){
	ComplexMatrix matrix(4,4);
	double theta = M_PI / 2.0;
	matrix << 1,0,0,0,
			  0,std::cos(theta),-1.0i*std::sin(theta),0,
			  0,-1.0i*std::sin(theta),std::cos(theta),0,
			  0,0,0,1;
	QuantumGateMatrix* tmp = gate::DenseMatrix(inputs,matrix);
	ans -> add_gate(tmp);	
}


void Google_random_circuit(int length,int depth,QuantumCircuit *ans){
	int n = length*length;
	for(int k = 0;k < depth;++k){
		for(int i = 0;i < n;++i){
			random_1Q_gate(i,ans);
		}
		const int dx[4] = {1,-1,0,0};
		const int dy[4] = {0,0,1,-1};
		for(int dir = 0;dir < 4;++dir){
			for(int i = 0;i < length;++i){
				for(int j = 0;j < length;++j){
					if((i+j) % 2 == 0){
						int x = i + dx[dir];
						int y = j + dy[dir];
						if(x >= 0 && x < length && y >= 0 && y < length){
							grid_2Q_gate({(UINT)i*length+j,(UINT)(x)*length+y},ans);
						}
					}
				}
			}
		}
	}
	return;
}

void Google_noisy_random_circuit(int length,int depth,QuantumCircuit *ans,double prob){
	int n = length*length;
	for(int k = 0;k < depth;++k){
		for(int i = 0;i < n;++i){
			random_1Q_gate(i,ans);
			ans -> add_gate(gate::DepolarizingNoise(i,prob));
		}
		const int dx[4] = {1,-1,0,0};
		const int dy[4] = {0,0,1,-1};
		for(int dir = 0;dir < 4;++dir){
			for(int i = 0;i < length;++i){
				for(int j = 0;j < length;++j){
					if((i+j) % 2 == 0){
						int x = i+dx[dir];
						int y = j + dy[dir];
						if(x >= 0 && x < length && y >= 0 && y < length ){
							grid_2Q_gate({(UINT)i*length+j,(UINT)(x)*length+y},ans);
							ans -> add_gate(gate::TwoQubitDepolarizingNoise(i*length+j,x*length+y,prob));
						}
					}
				}
			}
		}
	}
	return;
}


int main(){
	int n;
	double prob;
	int sample_count;
	printf("Input sqrt(n): ");
	scanf("%d",&n);
	n *= n;
	printf("Input noise prob: ");
	scanf("%lf",&prob);
	printf("Input sampling count: ");
	scanf("%d",&sample_count);
	{
		auto start = std::chrono::system_clock::now();
		QuantumCircuit circuit(n);
		Google_random_circuit(sqrt(n),10,&circuit);
        std::cout << circuit << std::endl;
		NoiseSimulator hoge(&circuit);
		auto A = hoge.execute(sample_count,prob);
		
		auto end = std::chrono::system_clock::now();
		auto dur = end - start;
		auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::cout << msec << std::endl;
	}
	{
    	auto start = std::chrono::system_clock::now();
		QuantumState state(n);
		state.set_zero_state();
		QuantumState base_state(n);
		base_state.set_zero_state();
		QuantumCircuit circuit(n);
		Google_noisy_random_circuit(sqrt(n),10,&circuit,0.01);
		std::vector<int> ans;
		for(int i = 0;i < sample_count;++i){
			state.load(&base_state);
			circuit.update_quantum_state(&state);
			ans.push_back(state.sampling(1)[0]);
		}
		auto end = std::chrono::system_clock::now();
		auto dur = end - start;
		auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::cout << msec << std::endl;
	}
}