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
mamba create -y -p $deps_dir c-compiler cxx-compiler cmake llvmdev \
    tbb-devel tbb libboost-devel 'mppp=1.*' sleef xtensor xtensor-blas \
    blas blas-devel fmt spdlog ninja
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# Clear the compilation flags set up by conda.
unset CXXFLAGS
unset CFLAGS

# Configure.
cmake -G Ninja ../ \
    -DCMAKE_PREFIX_PATH=$deps_dir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DHEYOKA_BUILD_TESTS=yes \
    -DHEYOKA_BUILD_TUTORIALS=ON \
    -DHEYOKA_WITH_MPPP=yes \
    -DHEYOKA_WITH_SLEEF=yes \
    -DCMAKE_CXX_FLAGS="-fsanitize=undefined" \
    -DCMAKE_CXX_FLAGS_DEBUG="-g -Og" \
    -DBoost_NO_BOOST_CMAKE=ON

# Build.
ninja -v

# Run the tests.
ctest -VV -j4

set +e
set +x
