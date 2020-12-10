#define _USE_MATH_DEFINES 
#include <mpi.h>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/noisesimulator.hpp>
#include <mpisim/noisesimulatorMPI.hpp>
#include <mpisim/utils.hpp>

double prob;

void random_1Q_gate(int qubitID,QuantumCircuit *val){
	int choice = rand() % 3LL;
	if(choice == 0){
		val -> add_noise_gate(gate::sqrtX(qubitID),"Depolarizing",prob);
		//val -> add_sqrtX_gate(qubitID);
	}else if(choice == 1){
		val -> add_noise_gate(gate::sqrtY(qubitID),"Depolarizing",prob);
		//val -> add_sqrtY_gate(qubitID);
	}else{
		ComplexMatrix matrix(2,2);
		matrix << 1.0/sqrt(2.0),-sqrt(1.0i/2.0),sqrt(-1.0i/2.0),1.0/sqrt(2.0);
		QuantumGateMatrix* tmp = gate::DenseMatrix(qubitID,matrix);
		//val -> add_gate(tmp);
		val -> add_noise_gate(tmp,"Depolarizing",prob);
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
	//ans -> add_gate(tmp);
	ans -> add_noise_gate(tmp,"Depolarizing",prob);	
}


void Google_random_circuit(int length,int depth,QuantumCircuit *ans){
	int n = length*length;
	for(int k = 0;k < depth;++k){
		const int dx[4] = {1,-1,0,0};
		const int dy[4] = {0,0,1,-1};
		for(int dir = 0;dir < 4;++dir){
			for(int i = 0;i < n;++i){
				random_1Q_gate(i,ans);
			}
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
		const int dx[4] = {1,-1,0,0};
		const int dy[4] = {0,0,1,-1};
		for(int dir = 0;dir < 4;++dir){
			for(int i = 0;i < n;++i){
				random_1Q_gate(i,ans);
				//ans -> add_gate(gate::DepolarizingNoise(i,prob));
			}
			for(int i = 0;i < length;++i){
				for(int j = 0;j < length;++j){
					if((i+j) % 2 == 0){
						int x = i+dx[dir];
						int y = j + dy[dir];
						if(x >= 0 && x < length && y >= 0 && y < length ){
							grid_2Q_gate({(UINT)i*length+j,(UINT)(x)*length+y},ans);
							//ans -> add_gate(gate::TwoQubitDepolarizingNoise(i*length+j,x*length+y,prob));
						}
					}
				}
			}
		}
	}
	return;
}

int main(int argc,char **argv){
	omp_set_num_threads(10);
    //printf("使用可能な最大スレッド数：%d\n", omp_get_max_threads());

	MPI_Init(&argc,&argv);
	int n = 16;
	int sample_count = 30000000;
	printf("%d\n",sample_count);
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(myrank == 0){
		printf("Input sqrt(n): \n");
		scanf("%d",&n);
		n *= n;
		printf("Input noise prob: \n");
		scanf("%lf",&prob);
		printf("Input sampling count: \n");
		scanf("%d",&sample_count);
	}
	
	MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&prob,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&sample_count,1,MPI_INT,0,MPI_COMM_WORLD);


	int seed = 0;
	
	{
		MPI_Barrier(MPI_COMM_WORLD);
		auto start = MPI_Wtime();
		srand(seed);
		QuantumCircuit circuit(n);
		Google_random_circuit(sqrt(n),5,&circuit);
		NoiseSimulatorMPI hoge(&circuit);
		auto A = hoge.execute(sample_count);
		auto end = MPI_Wtime();
		auto dur = end - start;
		auto msec = dur;//std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		if(myrank == 0){
			std::cout << "NoiseSimulatorMPI sec: " << msec << " sec"<< std::endl;
			for(int q = 0;q < A.size();q += std::max(1,(int)A.size())){
				std::cerr << A[q] << std::endl;
			}
		}
	}
	
	{
		MPI_Barrier(MPI_COMM_WORLD);
		auto start = MPI_Wtime();
		srand(seed);
		QuantumCircuit circuit(n);
		Google_random_circuit(sqrt(n),5,&circuit);
		NoiseSimulator hoge(&circuit);
		auto A = hoge.execute(sample_count/2);
		std::vector<UINT> final_ans;
		if(myrank == 0){
			Utility::receive_vector(0,1,0,final_ans);
		}else{
			Utility::send_vector(1,0,0,A);
		}
		auto end = MPI_Wtime();
		auto dur = end - start;
		auto msec = dur;//std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::cout << A.size() << std::endl;
		if(myrank == 0){
			std::cout << "NoiseSimulator sec: " << msec << " sec"<< std::endl;
		}
	}
	MPI_Finalize();
}