# About Qulacs

Qulacs is a Python/C++ library for fast simulation of large, noisy, or parametric quantum circuits.
Qulacs is developed at QunaSys, Osaka University, NTT, and Fujitsu.

Qulacs is licensed under the [MIT license](https://github.com/qulacs/qulacs/blob/master/LICENSE).

## Features

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

![multi thread benchmark](https://storage.googleapis.com/qunasys/fig_both_gpu.png)

## Requirement

- C++ compiler (gcc or VisualStudio)
    - gcc/g++ >= 7.0.0 (checked in Linux, MacOS, cygwin, MinGW, and WSL)
    - Microsoft VisualStudio C++ 2015 or 2017
- Python 2.7 or 3.x
- CMake >= 3.0
- Git
- (option) CUDA >= 8.0
- (option) AVX2 support

If your system supports AVX2 instructions, SIMD optimization is automatically enabled.
If you want to enable GPU simulator, install qulacs through `qulacs-gpu` package or build from source.
Note that `qulacs-gpu` includes a CPU simulator. You don't need to install both.

If you encounter some troubles, see {doc}`2_faq`.

### How to cite

Please cite this arXiv paper: [Qulacs: a fast and versatile quantum circuit simulator for research purpose](https://arxiv.org/abs/2011.13524).
