conda config --add channels conda-forge
conda config --set channel_priority strict
conda create --name heyoka
conda activate heyoka
conda install mamba -y
mamba install -y cmake llvmdev boost-cpp xtensor xtensor-blas blas blas-devel fmt
mkdir build
cd build
cmake ../ -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH=%CONDA_PREFIX_PATH% -DHEYOKA_BUILD_TESTS=yes -DHEYOKA_ENABLE_IPO=yes -DBoost_NO_BOOST_CMAKE=ON
cmake --build . --config Release
set PATH=%PATH%;%CD%\Release
echo %PATH%
ctest -V -C Release
