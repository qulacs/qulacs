#!/usr/bin/env bash

set -e

stubgen -p qulacs_osaka_core -o typings
pybind11-stubgen qulacs_osaka_core --no-setup-py --root-module-suffix='' --ignore-invalid=all --output-dir='./typings'
stubgen -p qulacs_osaka -o typings
pybind11-stubgen qulacs_osaka --no-setup-py --root-module-suffix='' --ignore-invalid=all --output-dir='./typings'
mkdir -p pysrc-withstubs
cp -R pysrc/* pysrc-withstubs/
cp -R typings/* pysrc-withstubs/
find pysrc-withstubs/ -name __init__.pyi | sed -e 's/__init__.pyi/py.typed/' | xargs touch
