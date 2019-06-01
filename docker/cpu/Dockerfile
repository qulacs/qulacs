FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    cmake \
    software-properties-common \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    gcc-7 \
    g++-7

RUN add-apt-repository ppa:jonathonf/python-3.6 -y && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.6 \
    python3.6-dev \
    python3.6-tk

RUN rm /usr/bin/python3 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/bin/python3 && \
    ln -s /usr/bin/pip3.6 /usr/bin/pip3.6

RUN pip3 install qulacs
