#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install build-essential wget

# Install conda+deps.
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -y -q -p $deps_dir cmake llvmdev boost-cpp mppp sleef xtensor xtensor-blas blas blas-devel fmt spdlog
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# GCC build.
cmake ../ -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_BUILD_TESTS=yes -DHEYOKA_BUILD_TUTORIALS=ON -DHEYOKA_WITH_MPPP=yes -DHEYOKA_WITH_SLEEF=yes -DCMAKE_CXX_FLAGS="--coverage" -DHEYOKA_ENABLE_IPO=yes -DBoost_NO_BOOST_CMAKE=ON
make -j2 VERBOSE=1
ctest -V -j2

# Upload coverage data.
bash <(curl -s https://codecov.io/bash) -x gcov-9

set +e
set +x
