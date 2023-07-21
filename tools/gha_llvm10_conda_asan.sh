#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O mambaforge.sh
export deps_dir=$HOME/local
export PATH="$HOME/mambaforge/bin:$PATH"
bash mambaforge.sh -b -p $HOME/mambaforge
mamba create -y -q -p $deps_dir c-compiler cxx-compiler cmake 'llvmdev=10.*' tbb-devel tbb boost-cpp 'mppp>=0.27' sleef xtensor xtensor-blas blas blas-devel 'fmt=9.*' spdlog
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# GCC build.
cmake ../ -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_BUILD_TESTS=yes -DHEYOKA_BUILD_TUTORIALS=ON -DHEYOKA_WITH_MPPP=yes -DHEYOKA_WITH_SLEEF=yes -DCMAKE_CXX_FLAGS="-fsanitize=address" -DBoost_NO_BOOST_CMAKE=ON
make -j2 VERBOSE=1
ctest -V -j2 -E vsop2013

set +e
set +x
