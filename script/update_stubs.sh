#!/usr/bin/env bash

set -e

stubgen -p qulacs_core -o typings
pybind11-stubgen qulacs_core --root-suffix '' --numpy-array-remove-parameters -o './typings'
stubgen -p qulacs -o typings
pybind11-stubgen qulacs --root-suffix '' --numpy-array-remove-parameters -o './typings'
cp -R typings/qulacs_core/* pysrc/qulacs/
find pysrc/ -name __init__.pyi | sed -e 's/__init__.pyi/py.typed/' | xargs touch

# format
black pysrc
isort pysrc
