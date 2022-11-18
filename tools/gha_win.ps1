echo "The conda prefix is: "
echo $env:CONDA_PREFIX
conda install heyoka cmake 'llvmdev=15.0.2' tbb-devel tbb boost-cpp xtensor xtensor-blas blas blas-devel fmt spdlog sleef zlib libzlib 'mppp>=0.27' -y
mkdir build
cd build
cmake ../ -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH=%CONDA_PREFIX_PATH% -DHEYOKA_BUILD_TESTS=yes -DHEYOKA_WITH_MPPP=yes -DHEYOKA_BUILD_TUTORIALS=ON -DHEYOKA_ENABLE_IPO=yes -DBoost_NO_BOOST_CMAKE=ON -DHEYOKA_WITH_SLEEF=yes
cmake --build . --config Release
copy Release\heyoka.dll test\Release\
ctest -j4 -V -C Release
