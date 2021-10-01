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
conda create -y -q -p $deps_dir c-compiler cxx-compiler cmake llvmdev tbb-devel tbb boost-cpp mppp sleef xtensor xtensor-blas blas blas-devel fmt spdlog sphinx sphinxcontrib-bibtex sphinx-book-theme
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# GCC build.
cmake ../ -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DHEYOKA_BUILD_TESTS=yes -DHEYOKA_BUILD_TUTORIALS=ON -DHEYOKA_WITH_MPPP=yes -DHEYOKA_WITH_SLEEF=yes -DCMAKE_CXX_FLAGS="-fsanitize=address" -DBoost_NO_BOOST_CMAKE=ON
make -j2 VERBOSE=1
ctest -V -j2 -E vsop2013

# Build the docs.
cd ../doc
export SPHINX_OUTPUT=`make html linkcheck 2>&1 >/dev/null`;
if [[ "${SPHINX_OUTPUT}" != "" ]]; then
    echo "Sphinx encountered some problem:";
    echo "${SPHINX_OUTPUT}";
    exit 1;
fi
echo "Sphinx ran successfully";

if [[ ! -z "${CI_PULL_REQUEST}" ]]; then
    echo "Testing a pull request, the generated documentation will not be uploaded.";
    exit 0;
fi

if [[ "${CIRCLE_BRANCH}" != "master" ]]; then
    echo "Branch is not master, the generated documentation will not be uploaded.";
    exit 0;
fi

# Check out the gh_pages branch in a separate dir.
cd ../
git config --global push.default simple
git config --global user.name "CircleCI"
git config --global user.email "bluescarni@gmail.com"
set +x
git clone "https://${GH_TOKEN}@github.com/bluescarni/heyoka.git" heyoka_gh_pages -q
set -x
cd heyoka_gh_pages
git checkout -b gh-pages --track origin/gh-pages;
git rm -fr *;
mv ../doc/_build/html/* .;
git add *;
# We assume here that a failure in commit means that there's nothing
# to commit.
git commit -m "Update Sphinx documentation, commit ${CIRCLE_SHA1} [skip ci]." || exit 0
PUSH_COUNTER=0
until git push -q
do
    git pull -q
    PUSH_COUNTER=$((PUSH_COUNTER + 1))
    if [ "$PUSH_COUNTER" -gt 3 ]; then
        echo "Push failed, aborting.";
        exit 1;
    fi
done

set +e
set +x
