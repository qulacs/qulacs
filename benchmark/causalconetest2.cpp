#include <vector>
#include <utility>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/type.hpp>
#include "../src/cppsim/causal_cone.hpp"

using namespace std;
int main(){

	for(UINT n = 1; n <= 2; n++){
		for(UINT m = 1; m <= 3; m++){
			for(UINT depth = 1; depth <= 4; depth++){
				QuantumState state((n * 2) * (m * 2));
				QuantumCircuit circuit((n * 2) * (m * 2));

				auto encode = [&](UINT h, UINT w)-> UINT{
					return h * m * 2 + w;
				};
				for(UINT d = 0; d < depth;){
					for(UINT k = 0; k < 2; k++){
						for(UINT i = 0; i < n; i++){
							for(UINT j = 0; j < m; j++){
								{
									std::vector<UINT> v(2);
									v[0] = encode(i * 2 + k, j * 2 + k);
									v[1] = encode(i * 2 + k, (j * 2 + 1 + k) % (m * 2));
									circuit.add_random_unitary_gate(v);
								}
								{
									std::vector<UINT> v(2);
									v[0] = encode(i * 2 + k, j * 2 + k);
									v[1] = encode((i * 2 + 1 + k) % (n * 2), j * 2 + k);
									circuit.add_random_unitary_gate(v);
								}

								{
									std::vector<UINT> v(2);
									v[0] = encode((i * 2 + 1 + k) % (n * 2), j * 2 + k);
									v[1] = encode((i * 2 + 1 + k) % (n * 2), (j * 2 + 1 + k) % (m * 2));
									circuit.add_random_unitary_gate(v);
								}
								{
									std::vector<UINT> v(2);
									v[0] = encode(i * 2 + k, (j * 2 + 1 + k) % (m * 2));
									v[1] = encode((i * 2 + 1 + k) % (n * 2), (j * 2 + 1 + k) % (m * 2));
									circuit.add_random_unitary_gate(v);
								}
							}
						}
						d++;
						if(d >= depth) break;
					}
				}
				Observable observable((n * 2) * (m * 2));
				mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
				for(UINT i = 0; i < (n * 2) * (m * 2); i++){
					string s; s+="Z "; s+= to_string(i);
					observable.add_operator(rnd(), s);
				}
	
				auto start = std::chrono::system_clock::now();
				circuit.update_quantum_state(&state);
				auto value = observable.get_expectation_value(&state);
				auto end = std::chrono::system_clock::now();

				double t1 = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
				start = std::chrono::system_clock::now();
				auto v2= CausalCone(circuit, observable);
				end = std::chrono::system_clock::now();
				double t2 = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
				//cout<<"value"<<" "<<value<<" "<<v2<<"\n";
				cout<< n * 2<<" "<<m * 2 <<" "<<depth * 4 <<" "<<t1<<" "<<t2<<"\n";
			}
		}
	}

	
	//cout<<v2<<" "<<t2<<endl;
}