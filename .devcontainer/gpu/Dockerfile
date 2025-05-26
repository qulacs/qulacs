FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

ENV PROJECT_DIR="/workspaces/qulacs"
# Add build artifact to PYTHONPATH and python can find qulacs.
# Egg file name might vary depending on qulacs and python version.
ENV PYTHONPATH="${PROJECT_DIR}/dist:${PYTHONPATH}"
ENV PYTHONPATH="${PROJECT_DIR}/build:${PYTHONPATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    clang-format-14 \
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
    python3-venv \
    wget \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/clang-format-14 /usr/bin/clang-format

RUN python3 -m venv /opt/venv

RUN /opt/venv/bin/pip install -U pip \
    && /opt/venv/bin/pip install black flake8 openfermion mypy pybind11-stubgen setuptools

ENV PATH="/opt/venv/bin:$PATH"
