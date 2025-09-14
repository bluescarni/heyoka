#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda create -y -p $deps_dir c-compiler zlib cxx-compiler libcxx cmake ninja \
    llvmdev tbb-devel tbb libboost-devel sleef xtensor xtensor-blas blas openssl \
    blas-devel fmt spdlog mppp
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# Clear the compilation flags set up by conda.
unset CXXFLAGS
unset CFLAGS

# Configure.
#
# NOTE: turn off tutorials until the fmt+malloc compilation failure is resolved.
CXX=clang++ CC=clang cmake -G Ninja ../ \
    -DCMAKE_PREFIX_PATH=$deps_dir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DHEYOKA_BUILD_TESTS=yes \
    -DHEYOKA_WITH_MPPP=yes \
    -DHEYOKA_BUILD_TUTORIALS=OFF \
    -DHEYOKA_WITH_SLEEF=yes \
    -DCMAKE_CXX_FLAGS_DEBUG="-g -Og" \
    -DHEYOKA_FORCE_STATIC_LLVM=yes \
    -DCMAKE_CXX_FLAGS="-fsanitize=address" \
    -DHEYOKA_HIDE_LLVM_SYMBOLS=yes

# Build.
ninja -v

# Run the tests.
#
# See: https://github.com/google/sanitizers/wiki/AddressSanitizerContainerOverflow
ASAN_OPTIONS=detect_container_overflow=0 ctest -VV -j4

set +e
set +x
