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
conda create -y -p $deps_dir c-compiler cxx-compiler cmake \
    llvmdev tbb-devel tbb libboost-devel mppp sleef xtensor \
    xtensor-blas blas blas-devel fmt spdlog ninja openssl \
    'sphinxcontrib-bibtex=2.6.*' 'sphinx=7.*' 'sphinx-book-theme=1.*'
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# Clear the compilation flags set up by conda.
unset CXXFLAGS
unset CFLAGS

# Configure.
cmake ../ -G Ninja \
    -DCMAKE_PREFIX_PATH=$deps_dir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DHEYOKA_BUILD_TESTS=yes \
    -DHEYOKA_BUILD_TUTORIALS=ON \
    -DHEYOKA_WITH_MPPP=yes \
    -DHEYOKA_WITH_SLEEF=yes \
    -DCMAKE_CXX_FLAGS="-fsanitize=address" \
    -DCMAKE_CXX_FLAGS_DEBUG="-g -Og"

# Build.
ninja -v -j4

# Run the tests.
ctest -VV -j4

# Build the docs.
cd ../doc
export SPHINX_OUTPUT=`make html linkcheck 2>&1 >/dev/null`;
if [[ "${SPHINX_OUTPUT}" != "" ]]; then
    echo "Sphinx encountered some problem:";
    echo "${SPHINX_OUTPUT}";
    exit 1;
fi
echo "Sphinx ran successfully";

set +e
set +x
