# Qulacs-Osaka

[![Ubuntu & Windows CI](https://github.com/Qulacs-Osaka/qulacs-osaka/actions/workflows/ci.yml/badge.svg)](https://github.com/Qulacs-Osaka/qulacs-osaka/actions/workflows/ci.yml)
[![Downloads](https://pepy.tech/badge/qulacs-osaka)](https://pepy.tech/project/qulacs-osaka)

Qulacs-Osaka is a python/C++ library for fast simulation of large, noisy, or parametric quantum circuits. This project is (implicitly) forked from [Qulacs](https://github.com/qulacs/qulacs) and developed at Osaka University. 

Qulacs-Osaka is licensed under the [MIT license](https://github.com/qulacs/qulacs/blob/master/LICENSE).

## Relation to [Qulacs](https://github.com/qulacs/qulacs)
The commits in Qulacs are merged into Qulacs-Osaka up to the pull request corresponding to the commit  `987474b31a6e60eba116d2e40ab538dcf8086038` ([link to the corresponding commit in Qulacs](https://github.com/qulacs/qulacs/commit/987474b31a6e60eba116d2e40ab538dcf8086038) and [in Qulacs-Osaka](https://github.com/qulacs/qulacs/commit/987474b31a6e60eba116d2e40ab538dcf8086038)). After that commit, this project is developed independently from Qulacs and has many new features that have not been available in Qulacs.


## Features
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
For each qubit number N:
- Apply simultaneous random single-qubit Pauli-X rotation  

and then repeat:
- Apply CNOT(i,(i+1)%N) for all i in [0..N-1]
- Apply simultaneous random single-qubit Pauli-X rotation  

for N times.

#### Note
- Execution time include time for creating quantum circuit
- Benchmark was done with float64 precision (qsim was done with float32)

### Single thread benchmark

![single thread benchmark](https://storage.googleapis.com/qunasys/singlethread_plot2.png)

### Multi thread / GPU benchmark

![multi thread benchmark](https://storage.googleapis.com/qunasys/multithread_plot2.png)

This benchmark was done with majour quantum circuit simulators with python interface.  
[Yao](https://github.com/QuantumBFS/Yao.jl) is quantum circuit simulator using Julia that is as fast as Qulacs.  
Benchmark inculde Yao can be found [here](https://github.com/Roger-luo/quantum-benchmarks/blob/master/RESULTS.md).  


## Requirements

- C++ compiler (gcc or VisualStudio)
  - gcc/g++ >= 7.0.0 (checked in Linux, MacOS, cygwin, MinGW, and WSL)
  - Microsoft VisualStudio C++ 2015 or later
- [Boost](https://github.com/boostorg/boost) >= 1.71.0 (Minimum version tested in CI)
- Python >= 3.7
- CMake >= 3.0
- git
- (option) CUDA >= 8.0
- (option) AVX2 support

If your system supports AVX2 instructions, SIMD optimization is automatically enabled. 
If you want to enable GPU simulator, install qulacs through `qulacs-gpu` package or build from source.
Note that `qulacs-gpu` includes CPU simulator. You don't need to install both.

Qulacs is tested on the following systems.

- Ubuntu 20.04
- macOS Big Sur 11
- Windows Server 2019


## Install via pip
You can install the Python package via pip.
```bash
pip install qulacs-osaka
```


## Install from Source
If you encounter some troubles, see [troubleshooting](http://qulacs.org/md_4__trouble_shooting.html).


### Install Python library from source

Install (Multi-thread without GPU)
```
python setup.py install
```

Install (Multi-thread with GPU. CUDA is required)
```
USE_GPU=Yes python setup.py install
```

Uninstall
```
pip uninstall qulacs
```

### Build C++ library

#### GCC
```
git clone https://github.com/Qulacs-Osaka/qulacs-osaka.git
cd qulacs-osaka
./script/build_gcc.sh
```

When you want to build with GPU, use `build_gcc_with_gpu.sh`.

#### MSVC
```
git clone https://github.com/Qulacs-Osaka/qulacs-osaka.git
cd qulacs-osaka
./script/build_msvc_2017.bat
```

When you want to build with GPU, use `build_msvc_2017_with_gpu.bat`. If you use MSVC2015, replace "2017" in file names to "2015".

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

If you want to run it on GPU, install GPU-enabled qulacs and replace `QuantumState` in the above codes to `QuantumStateGpu`.

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
g++ -O2 -I ./include -L ./lib main.cpp -fopenmp -lcppsim_static -lcsim_static
```

If you want to run it on GPU, include `cppsim/state_gpu.hpp` and replace `QuantumState` with `QuantumStateGpu`.

## How to cite

Please cite this arXiv paper: [Qulacs: a fast and versatile quantum circuit simulator for research purpose](https://arxiv.org/abs/2011.13524)
