#!/bin/bash

set -eux

mkdir -p coverage

lcov \
-d "build/src/csim/CMakeFiles/csim_static.dir" \
-d "build/src/cppsim/CMakeFiles/cppsim_static.dir" \
-d "build/src/cppsim_experimental/CMakeFiles/cppsim_exp_static.dir" \
-d "build/src/vqcsim/CMakeFiles/vqcsim_static.dir" \
-c \
-o coverage/coverage.info

lcov -r coverage/coverage.info "*/include/*" -o coverage/coverageFiltered.info

genhtml -o coverage/html --num-spaces 4 -s --legend coverage/coverageFiltered.info
