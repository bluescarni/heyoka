#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
mamba create -y -p $deps_dir c-compiler cxx-compiler ninja cmake \
    llvmdev tbb-devel tbb libboost-devel 'mppp=1.*' sleef xtensor xtensor-blas \
    blas blas-devel fmt spdlog zlib
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# Configure.
cmake ../ -G Ninja \
    -DCMAKE_PREFIX_PATH=$deps_dir \
    -DCMAKE_BUILD_TYPE=Release \
    -DHEYOKA_BUILD_TESTS=yes \
    -DHEYOKA_BUILD_TUTORIALS=ON \
    -DHEYOKA_WITH_MPPP=yes \
    -DHEYOKA_WITH_SLEEF=yes \
    -DHEYOKA_ENABLE_IPO=yes \
    -DHEYOKA_FORCE_STATIC_LLVM=yes \
    -DHEYOKA_HIDE_LLVM_SYMBOLS=yes

# Build.
ninja -v

# Run the tests.
ctest -VV -j4

set +e
set +x
