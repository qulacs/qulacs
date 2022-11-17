# FAQ

## Trouble-shootings

### Compile error in C/C++

#### When I try to compile the C++ library, I get compile errors.

Codes might not be compiled with `gcc` or `g++` version>=7.0.0.
Please check codes are compiled with `gcc` and `g++` and its version is greater or equal to 7.0.0.

For macOS Users: The default backend of `gcc` or `g++` command is `clang`, which is not supported in Qulacs.

#### When I compile Qulacs with MinGW, the compiler says header files such as `crypt.h` or `sys/select.h` was not found. When objects are linked, the linker says library `-lintl` was not found.

This may occur when you try to build the 32bit Python library withthe 64bit compiler,
or when you try to build the 64bit Python library with the 32bit compiler.
If you compile the C++ library with 32bit/64bit, the Python library must be 32bit/64bit, respectively.

### Error in Python library

#### I have many versions of Python, and I want to build Qulacs for a specific one.

Qulacs is built using the default Python.
Please set the version of Python you want to use with `pyenv` or `conda` as the default.
Then build Qulacs.

You can also specify a Python binary when building with CMake by adding `-D PYTHON_EXECUTABLE:FILEPATH=/usr/bin/pythonx.x`.

#### When I import the library, Python says there is no init function.

If you use Qulacs from Python and call functions directly using dll/pyd,
do not change the name of the Python library. 
If you have changed the dll/pyd name of the Python library, you will see this error.

If you import the dll/pyd built with a different version of Python, you may see this error.

#### When I import the library, I get a Segmentation fault error.
#### Why does Python immediately exit when I import the library?
#### Why does Python miss functions that start with `Py_`?
#### Though there exist dll files, Python says there is no dll/pyd.

If you import dll/pyd built with a different version of Python, you will see these errors.
Error messages depend on the version of Python.
