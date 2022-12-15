# See here for image contents: https://github.com/microsoft/vscode-dev-containers/blob/main/containers/python-3/.devcontainer/base.Dockerfile

# Pick a Python version: 3, 3.10, 3.9, 3.8, 3.7, 3.6
# Append -bullseye or -buster to pin to an OS version.
# Use -bullseye variants on local on arm64/Apple Silicon.
FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9-bullseye

ENV DEBIAN_FRONTEND=noninteractive

ENV PWD="/workspaces/qulacs"
ENV QULACS_PACKAGE="qulacs"
ENV USE_TEST="Yes"
# Add build artifact to PYTHONPATH and python can find qulacs.
# Egg file name might vary depending on qulacs and python version.
ENV PYTHONPATH="${PWD}/dist:${PYTHONPATH}"
ENV PYTHONPATH="${PWD}/build:${PYTHONPATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    clang-format \
    gdb \
    cmake \
    libboost-dev \
    doxygen \
    pandoc \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip \
    && pip install black flake8 mypy openfermion pybind11-stubgen

# Install cereal
RUN git clone https://github.com/USCiLab/cereal.git -b v1.3.0 --depth 1 /tmp/cereal \
    && cp -r /tmp/cereal/include/cereal /usr/local/include \
    && rm -rf /tmp/cereal

# Install Eigen
RUN git clone https://gitlab.com/libeigen/eigen.git -b 3.3.7 --depth 1 /tmp/eigen \
    && cp -r /tmp/eigen/Eigen /usr/local/include \
    && rm -rf /tmp/eigen

# Install Google Test
RUN git clone https://github.com/google/googletest.git -b release-1.8.1 --depth 1 /tmp/gtest \
    && cp -r /tmp/gtest/googletest/include/gtest /usr/local/include \
    && rm -rf /tmp/gtest

# Install pybind11
RUN git clone https://github.com/pybind/pybind11.git -b v2.10.0 --depth 1 /tmp/pybind11 \
    && cp -r /tmp/pybind11/include/pybind11 /usr/local/include \
    && rm -rf /tmp/pybind11

# Install lcov
RUN git clone https://github.com/linux-test-project/lcov.git -b v1.15 --depth 1 /tmp/lcov \
    && cd /tmp/lcov \
    && make install \
    && cd - \
    && rm -rf /tmp/lcov
