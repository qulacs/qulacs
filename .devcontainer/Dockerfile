# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version (use Debian 11/9, Ubuntu 18.04/21.04 on local arm64/Apple Silicon): debian-11, debian-10, debian-9, ubuntu-21.04, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHON_VERSION="3.9.5"
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
ENV PWD="/qulacs-osaka"
ENV QULACS_OSAKA_PACKAGE="qulacs_osaka"
ENV QULACS_OSAKA_VERSION="0.2.0"
# Add build artifact to PYTHONPATH and python can find qulacs.
# Egg file name might vary depending on qulacs and python version.
ENV PYTHONPATH="${PWD}/dist/${QULACS_OSAKA_PACKAGE}-${QULACS_OSAKA_VERSION}-cp38-cp38-linux_x86_64.whl:${PYTHONPATH}"
ENV PYTHONPATH="${PWD}/build/lib.linux-x86_64-3.8:${PYTHONPATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    clang-format \
    cmake \
    curl \
    gcc \
    git \
    gosu \
    g++ \
    libboost-dev \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    python3-dev \
    python3-distutils \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev

# Install pyenv
RUN git clone https://github.com/pyenv/pyenv.git ${PYENV_ROOT} \
    && pyenv install ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION} \
    && pyenv rehash \
    && python -m pip install --upgrade pip \
    && pip install mypy numpy pybind11-stubgen scipy wheel
