# Contributing to qulacs

Thank you for your interest in qulacs!

There are many ways to contribute not only for writing code.
This document provides a overview for contributing.
## Asking Questions
If you have questions to qulacs, feel free to ask in Issue and community will assist you! Your question will be a knowledge to others seeking for help.
Issue title should be prefixed with `[Question]` so that community can identify question.
## Report bugs
Reporting problems are always welcome! Post in Issue and we'll understand caveats.
Issue title should be prefixed with `[Bug]`.
Please include following details if you have:
- Reproducible steps to cause the issue
- What qulacs expected to do, and actually done
- code snippet to demonstorate issue
- Version of qulacs, OS info, etc...

## Create feature requests
Have a feature request you want? Post it in Issue!
Issue title should be prefixed with `[Feat]`.

## Pull Request

### Pull Request Process
PR is always welcome!

Writing to Issue is not necessary (but recommended) to create single PR. Just create PR to the `main` branch.

However, if you want to make changes that requires multiple PR, please discuss the change you wish to make via Issue.
### Size of Pull Request
When you want to create PR, non-automatically created diff should not be more than about 300 lines.

If you want to make large PR, then first ask to maintainers and feature branch will be created for review.
## Requirements to merge Pull Request

- Patch is legal under the LICENSE.
- Pass GitHub Actions' check.
- Approve of at least 1 maintainer.
## How to test

Testing locally is recommended before submitting PR.
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

- We recommend use Python from [pyenv](https://github.com/pyenv/pyenv) and [virtualenv](https://pypi.org/project/virtualenv/) for the local test.
    - Since we run `python setup.py install` at global Python unstable qulacs would be installed unintentionally.
    - And it would be difficult to show dependencies version when we need for debug
