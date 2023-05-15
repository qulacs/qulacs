#!/bin/env bash

set -e

find ./src ./test ./benchmark ./python -regex '.*\.\(cu\|cuh\|cpp\|h\|hpp\)' -exec clang-format -style=file -i {} \;