#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-aarch64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
mamba create -y -q -p $deps_dir cxx-compiler c-compiler cmake llvmdev tbb-devel tbb boost-cpp sleef xtensor xtensor-blas blas blas-devel fmt spdlog 'mppp>=0.27'
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# GCC build.
cmake ../ -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_BUILD_TESTS=yes -DHEYOKA_WITH_MPPP=yes -DHEYOKA_BUILD_TUTORIALS=ON -DHEYOKA_WITH_SLEEF=yes -DCMAKE_CXX_FLAGS="--coverage" -DBoost_NO_BOOST_CMAKE=ON
make -j2 VERBOSE=1
# Run the tests.
ctest -V -j2

# Upload coverage data.
bash <(curl -s https://codecov.io/bash) -x $deps_dir/bin/gcov

set +e
set +x
