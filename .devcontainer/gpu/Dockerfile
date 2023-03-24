FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

ENV PWD="/workspaces/qulacs"
ENV QULACS_PACKAGE="qulacs"
ENV USE_TEST="Yes"
# Add build artifact to PYTHONPATH and python can find qulacs.
# Egg file name might vary depending on qulacs and python version.
ENV PYTHONPATH="${PWD}/dist:${PYTHONPATH}"
ENV PYTHONPATH="${PWD}/build:${PYTHONPATH}"

# The base image is not created for devcontainer inherently, so we need to handle non-root user by ourselves here.

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    clang-format \
    cmake \
    curl \
    doxygen \
    gdb \
    libboost-dev \
    libpython3-dev \
    manpages \
    man-db \
    pandoc \
    python3 \
    python3-distutils \
    python3-pip \
    wget \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip \
    && pip install black flake8 openfermion mypy pybind11-stubgen
