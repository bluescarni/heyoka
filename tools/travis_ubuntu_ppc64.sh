#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
#apt-get update
#apt-get -y install wget
sudo yum -y install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-ppc64le.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda create -y -q -p $deps_dir cxx-compiler c-compiler cmake llvmdev tbb-devel tbb boost-cpp sleef xtensor xtensor-blas blas blas-devel fmt spdlog make
source activate $deps_dir

# Create the build dir and cd into it.
cd $HOME/heyoka
mkdir build
cd build

# GCC build.
cmake ../ -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_BUILD_TESTS=yes -DHEYOKA_BUILD_TUTORIALS=ON -DHEYOKA_WITH_SLEEF=yes -DCMAKE_CXX_FLAGS="--coverage" -DBoost_NO_BOOST_CMAKE=ON
make -j2 VERBOSE=1
# Run the tests.
ctest -V -j2 -E vsop2013

# Upload coverage data.
bash <(curl -s https://codecov.io/bash) -x gcov-9

set +e
set +x
