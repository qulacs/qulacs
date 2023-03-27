# Usage

See the following documents for more detail.

- {doc}`4.1_python_tutorial`
- {doc}`4.2_cpp_tutorial`
- {doc}`../guide/2.0_python_advanced`
- {doc}`../pyRef/index`
- {doc}`../api/cpp_library_root`

## C++ Libraries

Add `./<qulacs_path>/include/` to include path, and
`./<qulacs_path>/lib/` to library path.

Example of C++ code:

``` cpp
#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>

int main(){
    QuantumState state(3);
    state.set_Haar_random_state();

    QuantumCircuit circuit(3);
    circuit.add_X_gate(0);
    auto merged_gate = gate::merge(gate::CNOT(0,1),gate::Y(1));
    circuit.add_gate(merged_gate);
    circuit.add_RX_gate(1,0.5);
    circuit.update_quantum_state(&state);

    Observable observable(3);
    observable.add_operator(2.0, "X 2 Y 1 Z 0");
    observable.add_operator(-3.0, "Z 2");
    auto value = observable.get_expectation_value(&state);
    std::cout << value << std::endl;
    return 0;
}
```

### GCC

You can build your codes with the following gcc commands:

```
g++ -O2 -I ./<qulacs_path>/include -L ./<qulacs_path>/lib <your_code>.cpp -lvqcsim_static -lcppsim_static -lcsim_static -fopenmp
```

If you want to run your codes with GPU, include `cppsim/state_gpu.hpp` and use `QuantumStateGpu` instead of `QuantumState` and build with the following command:

```
nvcc -O2 -I ./<qulacs_path>/include -L ./<qulacs_path>/lib <your_code>.cu -lvqcsim_static -lcppsim_static -lcsim_static -lgpusim_static -D _USE_GPU -lcublas -Xcompiler -fopenmp
```

### MSVC

Your C++ codes can be built with Qulacs with the following process:

1. Create an empty project.
1. Select `x64` as an active solution platform.
1. Right Click your project name in Solution Explorer, and select `Properties`.
1. At `VC++ Directories` section, add the full path to `./qulacs/include` to `Include Directories`
1. At `VC++ Directories` section, add the full path to `./qulacs/lib` to `Library Directories`
1. At `C/C++` -\> `Code Generation` section, change `Runtime library` to `Multi-threaded (/MT)`.
1. At `Linker` -\> `Input` section, add `vqcsim_static.lib;cppsim_static.lib;csim_static.lib;` to `Additional Dependencies`.

## Python Libraries

You can use features by simply importing `qulacs`.

Example of Python code:

``` python
from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import Y,CNOT,merge

state = QuantumState(3)
state.set_Haar_random_state()

circuit = QuantumCircuit(3)
circuit.add_X_gate(0)
merged_gate = merge(CNOT(0,1),Y(1))
circuit.add_gate(merged_gate)
circuit.add_RX_gate(1,0.5)
circuit.update_quantum_state(state)

observable = Observable(3)
observable.add_operator(2.0, "X 2 Y 1 Z 0")
observable.add_operator(-3.0, "Z 2")
value = observable.get_expectation_value(state)
print(value)
```
