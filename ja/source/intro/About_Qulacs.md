 
# About Qulacs
[![Build Status](https://travis-ci.org/qulacs/qulacs.svg?branch=master)](https://travis-ci.org/qulacs/qulacs)

Qulacs is a fast quantum circuit simulator for simulating large, noisy, or parametric quantum circuits.

Qulacs is licensed under the [MIT license](https://github.com/qulacs/qulacs/blob/master/LICENSE).

## Feature
- Fast quantum circuit simulation with parallelized C/C++ backend
- Noisy quantum gate for simulation of NISQ devices
- Parametric quantum gates for variational methods
- Circuit compression for fast simulation
- GPU support for fast simulation
- Many utility functions for research

## Performance
- Compared following libraries on January, 2020

|       Package        | Version |
| -------------------- | ------- |
| [Qulacs GPU](https://github.com/qulacs/qulacs)     | 0.1.9   |
| [Cirq](https://github.com/quantumlib/Cirq)         | 0.6.0   |
| [Qiskit Aer](https://github.com/Qiskit/qiskit-aer) | 0.3.4   |
| [ProjectQ](https://github.com/ProjectQ-Framework/ProjectQ) | 0.4.2   |
| [qHiPSTER](https://github.com/intel/Intel-QS) | [latest master branch](https://github.com/intel/Intel-QS/tree/94e47c04b33ad51c4cb07feade48612d8690e425)   |
| [Python interface](https://github.com/HQSquantumsimulations/PyQuEST-cffi) of [QuEST](https://github.com/QuEST-Kit/QuEST) (PyQuest-cffi) | 3.0.0   |

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
   
   Note that measured time include time for create quantum circuit.

### Single thread benchmark


![single thread benchmark](https://storage.googleapis.com/qunasys/singlethread_plot.png)

### Multi thread / GPU benchmark



![multi thread benchmark](https://storage.googleapis.com/qunasys/multithread_plot.png)


This benchmark was done with majour quantum circuit simulator with python interface.  
[Yao](https://github.com/QuantumBFS/Yao.jl) is quantum circuit simulator using Julia that is as fast as Qulacs.  
Benchmark inculde Yao can be found [here](https://github.com/Roger-luo/quantum-benchmarks/blob/master/RESULTS.md).  


## Requirement

- C++ compiler (gcc or VisualStudio)
    - gcc/g++ >= 7.0.0 (checked in Linux, MacOS, cygwin, MinGW, and WSL)
    - Microsoft VisualStudio C++ 2015 or 2017
- python 2.7 or 3.x
- cmake >= 3.0
- git
- (option) CUDA >= 8.0
- (option) AVX2 support

If your system supports AVX2 instructions, SIMD optimization is automatically enabled. If you want to enable GPU simulator, install qulacs through <code>qulacs-gpu</code> package or build from source.

Qulacs is tested on the following systems.

- Ubuntu 16.04 / 18.04
- MacOS X Sierra
- Windows 10
