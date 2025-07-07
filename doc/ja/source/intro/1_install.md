# Installation

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

## Install Python library from source

To install Qulacs optimized for your system, we recommend the following install procedure for faster simulation of quantum circuits, while this requires a compiler and takes time for installation. In addition, you can enable or disable optimization features such as SIMD optimization, OpenMP parallelization, and GPU support.

A binary that is installed via pip command is optimized for Haswell architecture. Thus, Qulacs installed via pip command does not work with a CPU older than Haswell. If your CPU is newer than Haswell, Qualcs built from source shows the better performance.

### Requirements

- C++ compiler (gcc or VisualStudio)
  - gcc/g++ >= 7.0.0 (checked in Linux, MacOS, cygwin, MinGW, and WSL)
  - Microsoft VisualStudio C++ 2015 or later
- [Boost](https://github.com/boostorg/boost) >= 1.71.0 (Minimum version tested in CI)
- Python >= 3.9
- CMake >= 3.21
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

If you encounter some troubles, see {doc}`2_faq`.

### How to install

Install with default options (Multi-thread without GPU):

```
pip install .
```

If AVX2 instructions are not supported, SIMD optimization is automatically disabled.


Install with GPU support (CUDA is required):

```
USE_GPU=Yes pip install .
```

Install single-thread Qulacs:

```
USE_OMP=No pip install .
```

The number of threads used in Qulacs installed with default options can be controlled via the environment variable `OMP_NUM_THREADS` or `QULACS_NUM_THREADS`.
While `OMP_NUM_THREADS` affects the parallelization of other libraries, `QULACS_NUM_THREADS` controls only the parallelization of QULACS.
Or, if you want to force only Qulacs to use a single thread, You can install single-thread Qulacs with the above command.

For development purpose, optional dependencies can be installed as follows.
```
# Install development tools
pip install .[dev]
# Install dependencies for document generation
pip install .[doc]
```

Uninstall Qulacs:

```
pip uninstall qulacs
```

## Use Qulacs as C++ library

### Build with GCC

Static libraries of Qulacs can be built with the following commands:

```
git clone https://github.com/qulacs/qulacs.git
cd qulacs
./script/build_gcc.sh
```

To build shared libraries, execute `make shared` at `./qulacs/build` folder.
When you want to build with GPU, use `build_gcc_with_gpu.sh` instead of `build_gcc.sh`.

Then, you can build your codes with the following gcc command:

```sh
g++ -O2 -I ./<qulacs_path>/include -L ./<qulacs_path>/lib <your_code>.cpp -lvqcsim_static -lcppsim_static -lcsim_static -fopenmp
```

If you want to run your codes with GPU, include `cppsim/state_gpu.hpp` and use `QuantumStateGpu` instead of `QuantumState` and build with the following command:

```
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
1. Select "x64" as an active solution platform.
1. Right Click your project name in Solution Explorer, and select "Properties".
1. At "VC++ Directories" section, add the full path to `./qulacs/include` to "Include Directories"
1. At "VC++ Directories" section, add the full path to `./qulacs/lib` to "Library Directories"
1. At "C/C++ -> Code Generation" section, change "Runtime library" to "Multi-threaded (/MT)".
1. At "Linker -> Input" section, add `vqcsim_static.lib;cppsim_static.lib;csim_static.lib;` to "Additional Dependencies".
