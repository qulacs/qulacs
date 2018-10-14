
# Qulacs

Qulacs is a fast quantum circuit simulator for simulating large, noisy, or parametric quantum circuits.

Qulacs is licensed under the [MIT licence](https://github.com/qulacs/qulacs/blob/master/LICENSE).

## Install

If you encounter some troubles, see [troubleshootings](./).

### Suppoted systems
Qulacs is tested and supported on the following systems.

- Ubuntu 16.04
- MacOS X Sierra
- Windows 10

Python library is tested on python-2.7.15 and python-3.6.6.

### Requirements

- python 2.7 or 3.x
- gcc/g++ >= 7.0.0 or VisualStudio 2017
- cmake >= 2.8
- git

### C++ Library

#### GCC
```
git clone https://github.com/qulacs/qulacs.git
cd qulacs
./build_gcc.sh
```

#### MSVC
```
git clone https://github.com/qulacs/qulacs.git
cd qulacs
generate_msvc_project.bat
```
Then, open `Project.sln` in `./qulacs/visualstudio/`, and build all.

### Python Interface

Install
```
git clone https://github.com/qulacs/qulacs.git
cd qulacs
python setup.py install
```

Uninstall
```
pip uninstall qulacs
```

## Gettig started

See [Tutorial](./) and [API document](./) for more details.

### C++ Libraries

Add `./<qulacs_path>/include/` to include path, and `./<qulacs_path>/lib/` to library path. If you use dynamic link library, add `./<qulacs_path>/bin/` to library path instead.


Example of C++ code:
```cpp
#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/hamiltonian.hpp>

int main(){
    QuantumState state(3);
    state.set_Haar_random_state();

    QuantumCircuit circuit(3);
    circuit.add_X_gate(0);
    auto merged_gate = gate::merge(gate::CNOT(0,1),gate::Y(1));
    circuit.add_gate(merged_gate);
    circuit.add_RX_gate(1,0.5);
    circuit.update_quantum_state(&state);

    Hamiltonian ham(3)
    ham.add_operator(2.0, "X 2 Y 1 Z 0")
    ham.add_operator(-3.0, "Z 2")
    auto value = ham.get_expectation_value(&state)
    std::cout << value << std::endl;
    return 0;
}
```

Example of build command:
```sh
g++ -I ./<qulacs_path>/include -L ./<qulacs_path>/lib <your_code>.cpp -lcppsim.so
```

### Python Libraries
You can use features by simply importing `pycppsim`.

Example of python code:
```python
from pycppsim import Hamiltonian, QuantumCircuit, QuantumState
from pycppsim.gate import Y,CNOT,merge

state = QuantumState(3)
state.set_Haar_random_state()

circuit = QuantumCircuit(3)
circuit.add_X_gate(0)
merged_gate = merge(CNOT(0,1),Y(1))
circuit.add_gate(merged_gate)
circuit.add_RX_gate(1,0.5)
circuit.update_quantum_state(state)

ham = Hamiltonian(3)
ham.add_operator(2.0, "X 2 Y 1 Z 0")
ham.add_operator(-3.0, "Z 2")
value = ham.get_expectation_value(state)
print(value)
```
