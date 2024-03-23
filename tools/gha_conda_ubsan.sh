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
mamba create -y -q -p $deps_dir c-compiler cxx-compiler cmake llvmdev tbb-devel tbb libboost-devel 'mppp=1.*' sleef xtensor xtensor-blas blas blas-devel fmt spdlog ninja
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# GCC build.
cmake -G Ninja ../ -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_BUILD_TESTS=yes -DHEYOKA_BUILD_TUTORIALS=ON -DHEYOKA_WITH_MPPP=yes -DHEYOKA_WITH_SLEEF=yes -DCMAKE_CXX_FLAGS="-fsanitize=undefined" -DBoost_NO_BOOST_CMAKE=ON
ninja -j2 -v
ctest -V -j2 -E vsop2013

set +e
set +x
