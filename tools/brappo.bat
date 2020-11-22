"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create --name heyoka
conda activate heyoka
conda install mamba -y
mamba install -y cmake llvmdev boost-cpp xtensor xtensor-blas blas blas-devel fmt clang ninja
mkdir build
cd build
cmake ../ -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=%CONDA_PREFIX_PATH% -DHEYOKA_BUILD_TESTS=yes -DHEYOKA_ENABLE_IPO=yes -DBoost_NO_BOOST_CMAKE=ON
cmake --build . -- -v
