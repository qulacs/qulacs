 
# Qulacs
[![Build Status](https://travis-ci.org/qulacs/qulacs.svg?branch=master)](https://travis-ci.org/qulacs/qulacs)

Qulacs is a python/C++ library for fast simulation of large, noisy, or parametric quantum circuits.

Qulacs is licensed under the [MIT license](https://github.com/qulacs/qulacs/blob/master/LICENSE).

## Quick Install

```
pip install qulacs
```

## Feature
- Fast quantum circuit simulation with parallelized C/C++ backend
- Noisy quantum gate for simulation of NISQ devices
- Parametric quantum gates for variational methods
- Circuit compression for fast simulation
- GPU support for fast simulation
- Many utility functions for research

## Performance
- Compared processing time with following libraries on October 1st, 2018
    - Qulacs (ours)
    - [Cirq](https://github.com/quantumlib/Cirq)
    - [ProjectQ](https://github.com/ProjectQ-Framework/ProjectQ)
    - [pyQuil](https://github.com/rigetticomputing/pyquil)
    - [Q#](https://github.com/Microsoft/Quantum)
    - [Qiskit Terra QASM Simulator](https://github.com/Qiskit/qiskit-terra/tree/master/src/qasm-simulator-cpp)
    - [QuPy CPU & GPU](https://github.com/ken-nakanishi/qupy)

- Test environment:
    - 100 shot sampling of 10 layers of all random rotation X gate and 9 layers of all neighboring CNOT
    - Intel Core i7-8700 CPU
    - NVIDIA GTX 1050 Ti GPU
    - OpenMP enabled
    - MKL enabled (numpy runs in multi thread)
    - Circuit compression disabled
    
![benchmark](https://storage.googleapis.com/qunasys/_plot.png)

## Supported environment
Qulacs is tested on the following systems.

- OS
  - Ubuntu 16.04
  - MacOS X Sierra
  - Windows 10

The following languages are supported.

- C++
  - gcc/g++ >= 7.0.0
  - Microsoft VisualStudio C++ 2015 and 2017
- python
  - python 2.7
  - python 3.x


## Install from Source
If you encounter some troubles, see [troubleshooting (Japanese)](http://qulacs.org/md_4__trouble_shooting.html).
Currently, if you want to use GPU, qulacs must be installed from source.

### Requirements

- gcc/g++ >= 7.0.0 or Microsoft VisualStudio C++ 2017
- python 2.7 or 3.x
- cmake >= 3.0

### Install qulacs from source

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

### Build C++ and python library

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
script/build_msvc.bat
```

## Tutorial and API document

See the following documents for more detail.

- [Python Tutorial](http://qulacs.org/md_3__tutorial_python.html)
- [C++ Tutorial](http://qulacs.org/md_2__tutorial__c_p_p.html)  
- [Examples](https://github.com/qulacs/quantum-circuits)  
- [API document](http://qulacs.org/annotated.html)   

## Sample code
### Python
```python
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

If you want to run it on GPU, install qulacs from source and replace <code>QuantumState</code> with <code>QuantumStateGPU</code>.

### C++

```cpp
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

Build command for g++:
```sh
g++ -I ./<qulacs_path>/include -L ./<qulacs_path>/lib <your_code>.cpp -lcppsim.so
```

If you want to run it on GPU, include <code>cppsim/state_gpu.hpp</code> and replace <code>QuantumState</code> with <code>QuantumStateGPU</code>.


