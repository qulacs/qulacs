# Contributing

## Pull Request Process

When you want to contribute to this project,
please first discuss the change you wish to make via GitHub Issue.

1. Check that your patch is legal under the [LICENCE](https://github.com/qulacs/qulacs/blob/master/LICENSE).
2. Test in your computer.
    - If you want to know how to do it please see `How to test` section below.
3. PR to the `dev` instead of `master` branch.
    - Because we use `master` branch as the release branch.

## How to test

You should test in your computer before you submit a PR.
First you have to install [dependecies](https://github.com/qulacs/qulacs#requirement) to your computer
before you build and test your patch.
And you execute these commands at the root directory of this project:

### For macOS and Linux

```console
$ ./script/build_gcc.sh
$ python setup.py install
$ cd build
$ make test
$ make pythontest
```

### For Windows

```console
$ ./script/build_msvc_2015.bat
$ python setup.py install
$ cmake --build ./visualstudio --target test --config Release
$ cmake --build ./visualstudio --target pythontest --config Release
```
  
### Tips

- We recommend use Python from [pyenv](https://github.com/pyenv/pyenv) and [vertualenv](https://pypi.org/project/virtualenv/) for the local test.
    - Since we run `python setup.py install` at global Python unstable qulacs would be installed unintentionally.
    - And it would be difficult to show dependencies version when we need for debug
