Installation
==============


Requirements
~~~~~~~~~~~~

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


Quick Install
~~~~~~~~~~~~~~~

::

   pip install qulacs

If you have NVIDIA GPU with CUDA installed try:

::

   pip install qulacs-gpu


Install from Source
-------------------

If you encounter some troubles, see `troubleshooting`_.

Install python libs from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install (Multi-thread without GPU)

::

   python setup.py install

Install (Multithread with GPU. CUDA is required)

::

   python setup_gpu.py install

Install (Single-thread without GPU. For launching multiple qulacs
processes.)

::

   python setup_singlethread.py install

Uninstall

::

   pip uninstall qulacs

.. _build-c++-and-python-library:

Build C++ and python library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GCC
^^^

::

   git clone https://github.com/qulacs/qulacs.git
   cd qulacs
   ./script/build_gcc.sh

When you want to build with GPU, use build_gcc_with_gpu.sh.

MSVC
^^^^

::

   git clone https://github.com/qulacs/qulacs.git
   cd qulacs
   script/build_msvc_2017.bat

When you want to build with GPU, use build_msvc_2017_with_gpu.bat. If
you use MSVC2015, replace 2017 in file names to 2015.

