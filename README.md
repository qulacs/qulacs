# Qulacs

![CI](https://github.com/qulacs/qulacs/workflows/CI/badge.svg)
[![Downloads](https://pepy.tech/badge/qulacs)](https://pepy.tech/project/qulacs)

Qulacs is a Python/C++ library for fast simulation of large, noisy, or parametric quantum circuits.

Qulacs is licensed under the [MIT license](https://github.com/qulacs/qulacs/blob/master/LICENSE).

## Quick Install for Python

```
pip install qulacs
```

If your CPU is older than Intel Haswell architecture, the binary installed with the above command does not work. In this case, please install Qulacs with the following command.
Even if your CPU is newer than Haswell, Qulacs installed with the below command shows better performance but takes a longer time. See "Install Python library from source" section for detail.

```
pip install git+https://github.com/qulacs/qulacs.git
```

If you have NVIDIA GPU and CUDA is installed, GPU-version can be installed with the following command:
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

The time for simulating random quantum circuits is compared with several quantum circuit simulators in November 2020.

See [the benchmark repository](https://github.com/qulacs/benchmark-qulacs) and [Section VI and VII of our paper](https://arxiv.org/abs/2011.13524) for the detail of this benchmark.

Note that the plots with names ending with "opt" and "heavy opt" perform circuit optimization for fast simulation, where the time for optimization is included in the execution time.


### Single-thread benchmark

![single thread benchmark](https://storage.googleapis.com/qunasys/fig_both_singlethread.png)

### Multi-thread benchmark

![multi thread benchmark](https://storage.googleapis.com/qunasys/fig_both_multithread.png)

### GPU benchmark

![GPU benchmark](https://storage.googleapis.com/qunasys/fig_both_gpu.png)



## Tutorial and Example
### Documents

See the following documents for tutorials.

- [Python Tutorial](http://docs.qulacs.org/en/latest/intro/4.1_python_tutorial.html)
- [C++ Tutorial](http://docs.qulacs.org/en/latest/intro/4.2_cpp_tutorial.html)
- [Manual](http://docs.qulacs.org/en/latest/guide/2.0_python_advanced.html)
- [Examples](http://docs.qulacs.org/en/latest/apply/0_overview.html)  


### Python sample code
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

If you want to run it on GPU, install GPU-enabled qulacs and replace `QuantumState` in the above codes with `QuantumStateGpu`.

### C++ sample code

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


## Install Python library from source

To install Qulacs optimized for your system, we recommend the following install procedure for faster simulation of quantum circuits, while this requires a compiler and takes time for installation. In addition, you can enable or disable optimization features such as SIMD optimization, OpenMP parallelization, and GPU support.

A binary that is installed via pip command is optimized for Haswell architecture. Thus, Qulacs installed via pip command does not work with a CPU older than Haswell. If your CPU is newer than Haswell, Qualcs built from source shows the better performance.

### Requirement

- C++ compiler (gcc or VisualStudio)
    - gcc/g++ >= 7.0.0 (checked in Linux, MacOS, cygwin, MinGW, and WSL)
    - Microsoft VisualStudio C++ 2015 or later
- Python 2.7 or 3.x
- cmake >= 3.0
- git
- (option) CUDA >= 8.0
- (option) AVX2 support

If your system supports AVX2 instructions, SIMD optimization is automatically enabled. 
If you want to enable GPU simulator, install qulacs through `qulacs-gpu` package or build from source.
Note that `qulacs-gpu` includes a CPU simulator. You don't need to install both.

If you encounter some troubles, see [troubleshooting](http://qulacs.org/md_4__trouble_shooting.html).

### How to install

Install with default options (Multi-thread without GPU):
```
python setup.py install
```
If AVX2 instructions are not supported, SIMD optimization is automatically disabled.


Install with GPU support (CUDA is required):
```
python setup_gpu.py install
```


Install single-thread Qulacs:
```
python setup_singlethread.py install
```
The number of threads used in Qulacs installed with default options can be controlled via the environment variable `OMP_NUM_THREADS`.
However, typically this option also affects the parallelization of other libraries. 
If you want to force only Qulacs to use a single thread, You can install single-thread Qulacs with the above command.


Uninstall Qulacs:
```
pip uninstall qulacs
```

## Use Qualcs as C++ library

### Build with GCC
Static libraries of Qulacs can be built with the following command:
```
git clone https://github.com/qulacs/qulacs.git
cd qulacs
./script/build_gcc.sh
```

To build shared libraries, execute `make shared` at `./qulacs/build` folder.
When you want to build with GPU, use `build_gcc_with_gpu.sh` instead of `build_gcc.sh`.

Then, you can build your codes with the following gcc commands:
```sh
g++ -O2 -I ./<qulacs_path>/include -L ./<qulacs_path>/lib <your_code>.cpp -lvqcsim_static -lcppsim_static -lcsim_static -fopenmp
```

If you want to run your codes with GPU, include `cppsim/state_gpu.hpp` and use `QuantumStateGpu` instead of `QuantumState` and build with the following command:
```sh
nvcc -O2 -I ./<qulacs_path>/include -L ./<qulacs_path>/lib <your_code>.cu -lvqcsim_static -lcppsim_static -lcsim_static -lgpusim_static -D _USE_GPU -lcublas -Xcompiler -fopenmp
```

### Build with MSVC
Static libraries of Qulacs can be built with the following command:
```
git clone https://github.com/qulacs/qulacs.git
cd qulacs
script/build_msvc_2017.bat
```
When you want to build with GPU, use `build_msvc_2017_with_gpu.bat`.
If you use MSVC with other versions, use `build_msvc_2015.bat` or edit the generator name in `build_msvc_2017.bat`.

Your C++ codes can be built with Qulacs with the following process:

1. Create an empty project.
2. Select "x64" as an active solution platform.
3. Right Click your project name in Solution Explorer, and select "Properties".
4. At "VC++ Directories" section, add the full path to `./qulacs/include` to "Include Directories"
5. At "VC++ Directories" section, add the full path to `./qulacs/lib` to "Library Directories"
6. At "C/C++ -> Code Generation" section, change "Runtime library" to "Multi-threaded (/MT)".
7. At "Linker -> Input" section, add `vqcsim_static.lib;cppsim_static.lib;csim_static.lib;` to "Additional Dependencies".

## How to cite

Please cite this arXiv paper: [Qulacs: a fast and versatile quantum circuit simulator for research purpose](https://arxiv.org/abs/2011.13524)
