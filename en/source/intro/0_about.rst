About Qulacs
============


Qulacs is a fast quantum circuit simulator for simulating large, noisy,
or parametric quantum circuits.

Qulacs is licensed under the `MIT license`_.

Feature
-------

-  Fast quantum circuit simulation with parallelized C/C++ backend
-  Noisy quantum gate for simulation of NISQ devices
-  Parametric quantum gates for variational methods
-  Circuit compression for fast simulation
-  GPU support for fast simulation
-  Many utility functions for research

Performance
-----------

-  Compared following libraries on January, 2020

============================================== =======================
Package                                        Version
============================================== =======================
`Qulacs GPU`_                                  0.1.9
`Cirq`_                                        0.6.0
`Qiskit Aer`_                                  0.3.4
`ProjectQ`_                                    0.4.2
`qHiPSTER`_                                    `latest master branch`_
`Python interface`_ of `QuEST`_ (PyQuest-cffi) 3.0.0
============================================== =======================

Test environment:
~~~~~~~~~~~~~~~~~

-  Azure NC6s_v3 (6vcpu / Mem112GiB)
-  Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
-  Tesla V100 PCIE (driver 440.33.01)

What is Benchmarked
~~~~~~~~~~~~~~~~~~~

for each qubit number N:

-  Apply simultaneous random single-qubit Pauli-X rotation

and then repeat:

-  Apply CNOT(i,(i+1)%N) for all i in [0..N-1]
-  Apply simultaneous random single-qubit Pauli-X rotation

for N times.

Note that measured time include time for create quantum circuit.

Single thread benchmark
~~~~~~~~~~~~~~~~~~~~~~~

|single thread benchmark|

.. _multi-thread--gpu-benchmark:

.. |single thread benchmark| image:: https://storage.googleapis.com/qunasys/singlethread_plot.png




Multi thread / GPU benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|multi thread benchmark|

| This benchmark was done with major quantum circuit simulator with
  python interface.
| `Yao`_ is quantum circuit simulator using Julia that is as fast as
  Qulacs.
| Benchmark include Yao can be found `here`_.

Requirement
-----------

-  C++ compiler (gcc or VisualStudio)

   -  gcc/g++ >= 7.0.0 (checked in Linux, MacOS, cygwin, MinGW, and WSL)
   -  Microsoft VisualStudio C++ 2015 or 2017

-  python 2.7 or 3.x
-  cmake >= 3.0
-  git
-  (option) CUDA >= 8.0
-  (option) AVX2 support

If your system supports AVX2 instructions, SIMD optimization is
automatically enabled. If you want to enable GPU simulator, install
qulacs through qulacs-gpu package or build from source.

Qulacs is tested on the following systems.

-  Ubuntu 16.04 / 18.04
-  MacOS X Sierra
-  Windows 10

.. _MIT license: https://github.com/qulacs/qulacs/blob/master/LICENSE
.. _Qulacs GPU: https://github.com/qulacs/qulacs
.. _Cirq: https://github.com/quantumlib/Cirq
.. _Qiskit Aer: https://github.com/Qiskit/qiskit-aer
.. _ProjectQ: https://github.com/ProjectQ-Framework/ProjectQ
.. _qHiPSTER: https://github.com/intel/Intel-QS
.. _latest master branch: https://github.com/intel/Intel-QS/tree/94e47c04b33ad51c4cb07feade48612d8690e425
.. _Python interface: https://github.com/HQSquantumsimulations/PyQuEST-cffi
.. _QuEST: https://github.com/QuEST-Kit/QuEST
.. _Yao: https://github.com/QuantumBFS/Yao.jl
.. _here: https://github.com/Roger-luo/quantum-benchmarks/blob/master/RESULTS.md



.. |Build Status| image:: https://travis-ci.org/qulacs/qulacs.svg?branch=master
   :target: https://travis-ci.org/qulacs/qulacs
.. |single thread benchmark| image:: https://storage.googleapis.com/qunasys/singlethread_plot2.png
.. |multi thread benchmark| image:: https://storage.googleapis.com/qunasys/multithread_plot2.png
