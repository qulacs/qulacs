#!/usr/bin/env sh

# This script downloads the CMake binary and installs it in ~/.local directory
# (the cmake executable will be in ~/.local/bin).
# This is mostly suitable for CIs, not end users.

set -e

VERSION=3.12.3
PREFIX=~/.local

OS=$(uname -s)

BIN=$PREFIX/bin

if test -f $BIN/cmake && ($BIN/cmake --version | grep -q "$VERSION"); then
    echo "CMake $VERSION already installed in $BIN"
else
    FILE=cmake-$VERSION-$OS-x86_64.tar.gz
    URL=https://cmake.org/files/v3.12/$FILE
    ERROR=0
    TMPFILE=$(mktemp --tmpdir cmake-$VERSION-$OS-x86_64.XXXXXXXX.tar.gz)
    echo "Downloading CMake ($URL)..."
    wget "$URL" -O "$TMPFILE" -nv
    mkdir -p "$PREFIX"
    tar xzf "$TMPFILE" -C "$PREFIX" --strip 1
    rm $TMPFILE
fi
