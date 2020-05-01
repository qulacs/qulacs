 
# Qulacs
[![Build Status](https://travis-ci.org/qulacs/qulacs.svg?branch=master)](https://travis-ci.org/qulacs/qulacs)
[![Downloads](https://pepy.tech/badge/qulacs)](https://pepy.tech/project/qulacs)

Qulacs is a python/C++ library for fast simulation of large, noisy, or parametric quantum circuits.

Qulacs is licensed under the [MIT license](https://github.com/qulacs/qulacs/blob/master/LICENSE).

## Quick Install

```
pip install qulacs
```

If you have NVIDIA GPU with CUDA installed try:
```
pip install qulacs-gpu
```

## Feature
- Fast quantum circuit simulation with parallelized C/C++ backend
- Noisy quantum gate for simulation of NISQ devices
- Parametric quantum gates for variational methods
- Circuit compression for fast simulation
- GPU support for fast simulation
- Many utility functions for research

## Performance
- Compared following libraries on March, 2020

|       Package        | Version |
| -------------------- | ------- |
| [Qulacs GPU](https://github.com/qulacs/qulacs)     | 0.1.9   |
| [Cirq](https://github.com/quantumlib/Cirq)         | 0.6.0   |
| [Qiskit Aer](https://github.com/Qiskit/qiskit-aer) | 0.3.4   |
| [ProjectQ](https://github.com/ProjectQ-Framework/ProjectQ) | 0.4.2   |
| [qHiPSTER](https://github.com/intel/Intel-QS) | [latest master branch](https://github.com/intel/Intel-QS/tree/94e47c04b33ad51c4cb07feade48612d8690e425)   |
| [Python interface](https://github.com/HQSquantumsimulations/PyQuEST-cffi) of [QuEST](https://github.com/QuEST-Kit/QuEST) (PyQuest-cffi) | 3.0.0   |
| [qsim](https://github.com/quantumlib/qsim) | [latest master branch](https://github.com/quantumlib/qsim/tree/24a9af400c3d9e4aac011cb8e5dc6b9e1ac4233b)   |

### Test environment:
- Azure NC6s_v3 (6vcpu / Mem112GiB)
  - Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
  - Tesla V100 PCIE (driver 440.33.01)

### What is Benchmarked
   for each qubit number N:
   - Apply simultaneous random single-qubit Pauli-X rotation  
   
   and then repeat:
   - Apply CNOT(i,(i+1)%N) for all i in [0..N-1]
   - Apply simultaneous random single-qubit Pauli-X rotation  
   
   for N times.
   
#### Note
 - execution time include time for creating quantum circuit
 - benchmark was done with float64 precision (qsim was done with float32)

### Single thread benchmark

![single thread benchmark](https://storage.googleapis.com/qunasys/singlethread_plot2.png)

### Multi thread / GPU benchmark

![multi thread benchmark](https://storage.googleapis.com/qunasys/multithread_plot2.png)

This benchmark was done with majour quantum circuit simulators with python interface.  
[Yao](https://github.com/QuantumBFS/Yao.jl) is quantum circuit simulator using Julia that is as fast as Qulacs.  
Benchmark inculde Yao can be found [here](https://github.com/Roger-luo/quantum-benchmarks/blob/master/RESULTS.md).  


## Requirement

- C++ compiler (gcc or VisualStudio)
    - gcc/g++ >= 7.0.0 (checked in Linux, MacOS, cygwin, MinGW, and WSL)
    - Microsoft VisualStudio C++ 2015 or later
- python 2.7 or 3.x
- cmake >= 3.0
- git
- (option) CUDA >= 8.0
- (option) AVX2 support

If your system supports AVX2 instructions, SIMD optimization is automatically enabled. 
If you want to enable GPU simulator, install qulacs through <code>qulacs-gpu</code> package or build from source.
Note that <code>qulacs-gpu</code> includes CPU simulator. You don't need to install both.

Qulacs is tested on the following systems.

- Ubuntu 16.04 / 18.04
- MacOS X Sierra
- Windows 10


## Install from Source
If you encounter some troubles, see [troubleshooting](http://qulacs.org/md_4__trouble_shooting.html).


### Install python libs from source

Install (Multi-thread without GPU)
```
python setup.py install
```

Install (Multithread with GPU. CUDA is required)
```
python setup_gpu.py install
```

Install (Single-thread without GPU. For launching multiple qulacs processes.)
```
python setup_singlethread.py install
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
./script/build_gcc.sh
```

When you want to build with GPU, use <code>build_gcc_with_gpu.sh</code>.

#### MSVC
```
git clone https://github.com/qulacs/qulacs.git
cd qulacs
script/build_msvc_2017.bat
```

When you want to build with GPU, use <code>build_msvc_2017_with_gpu.bat</code>. If you use MSVC2015, replace "2017" in file names to "2015".

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

If you want to run it on GPU, install GPU-enabled qulacs and replace <code>QuantumState</code> in the above codes to <code>QuantumStateGpu</code>.

### C++

```cpp
#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

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
g++ -O2 -I ./<qulacs_path>/include -L ./<qulacs_path>/lib <your_code>.cpp -fopenmp -lcppsim_static.so
```

If you want to run it on GPU, include <code>cppsim/state_gpu.hpp</code> and replace <code>QuantumState</code> with <code>QuantumStateGpu</code>.

## How to cite
Please cite this GitHub URL: https://github.com/qulacs/qulacs

in bibtex style: 
```
@misc{Qulacs,
title = {{Qulacs}},
year = {2018},
eprint = "https://github.com/qulacs/qulacs"
}
```
