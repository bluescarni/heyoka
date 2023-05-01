#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -y -q -p $deps_dir cmake c-compiler cxx-compiler clang clangxx clang-tools llvmdev tbb-devel tbb boost-cpp 'mppp>=0.27' sleef fmt spdlog ninja
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# GCC build.
cmake ../ -G Ninja -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_WITH_MPPP=yes -DHEYOKA_WITH_SLEEF=yes -DBoost_NO_BOOST_CMAKE=ON \
    -DCMAKE_CXX_CLANG_TIDY=`which clang-tidy` -DCMAKE_C_CLANG_TIDY=`which clang-tidy` -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
ninja -j2

set +e
set +x
