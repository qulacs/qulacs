

# Troubleshootings

## Compile error in C/C++

### Error occurs when compiling csim_shared
Codes might not be compiled with gcc/g++ ver>=7.
Please check codes are compiled with gcc and g++ and it's version is greater or equal to 7.0.0.

For MacOS Users: the default backend of gcc/g++ command is clang, which is not supported in Qulacs.

### When we compile Qulacs with mingw, compiler says header files such as crypt.h or sys/select.h was not found. When objects are linked, linker says library -lintl was not found.

This may occur when you try to build 32bit python library with 64bit compiler, or 64bit lib with 32bit.
When you compile C++ with 32bit/64bit, the python library must be 32bit/64bit, respectively.

## Error in python library

### I have many versions of python, and want to build Qulacs for specific one.

Qulacs is build for default python and python-config. Please set the version where you want to install Qulacs as a default using pyenv or conda.
We can also specify python binary when we do cmake by adding -D PYTHON_EXECUTABLE:FILEPATH=/usr/bin/pythonx.x.

## When I import library, python says there is no init function.
If you use Qulacs from python and call functions directly using dll/pyd, the name of python library must not be changed. 
If you change the dll/pyd name of python library, you will see this error.

If you import python dll/pyd which is build for different python version, you may see this error.

### Segmentation fault occurs when I import library. Python immediately exit after importing library. Python says functions starting with "Py_" was not found. Though there exists dll files, python says there is no dll/pyd. 

If you import python dll/pyd which is build for different version python, you see these errors.
Error messages depend on the python version.



