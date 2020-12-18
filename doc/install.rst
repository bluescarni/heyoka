.. _installation:

Installation
============

Introduction
------------

heyoka is written in modern C++, and it requires a compiler able to understand
at least C++17. The library is regularly tested on
a comprehensive continuous integration pipeline, which includes:

* various versions of the three major compilers (GCC, Clang and MSVC),
* various versions of the three major operating systems
  (Linux, Windows and OSX).

heyoka has the following **mandatory** dependencies:

* the `LLVM <https://llvm.org/>`__ compiler infrastructure library, version 10 or 11,
* the `Boost <https://www.boost.org/>`__ C++ libraries (version >= 1.60),
* the `{fmt} <https://fmt.dev/latest/index.html>`__ library.

Additionally, heyoka has the following **optional** dependencies:

* the `mp++ <https://bluescarni.github.io/mppp/>`__ multiprecision library
  (provides support for quadruple-precision integrations),
* the `SLEEF <https://sleef.org/>`__ vectorized math library (improves the performance
  of integrations in batch mode),
* the `xtensor and xtensor-blas <https://xtensor.readthedocs.io/en/latest/>`__
  libraries (used in the tests and benchmarks).

`CMake <https://cmake.org/>`__ is the build system used by heyoka and it must also be available when
installing from source (the minimum required version is 3.8).

Installation from source
------------------------

Source releases of heyoka can be downloaded from
`github <https://github.com/bluescarni/heyoka/releases>`__.
Once in the source tree
of heyoka, you can use ``cmake`` to configure the build to your liking
(e.g., enabling optional features, customizing the installation
path, etc.). The available configuration options are:

* ``HEYOKA_WITH_MPPP``: enable features relying on the mp++ library (off by default),
* ``HEYOKA_WITH_SLEEF``: enable features relying on the SLEEF library (off by default),
* ``HEYOKA_BUILD_TESTS``: build the test suite (off by default),
* ``HEYOKA_BUILD_BENCHMARKS``: build the benchmarking suite (off by default),
* ``HEYOKA_BUILD_TUTORIALS``: build the tutorials (off by default),
* ``HEYOKA_BUILD_STATIC_LIBRARY``: build heyoka as a static library, instead
  of a dynamic library (off by default),
* ``HEYOKA_ENABLE_IPO``: enable link-time optimisations when building
  the heyoka library (requires CMake >= 3.9 and compiler support,
  off by default).

In order to build heyoka, you can run the following CMake command from the
build directory:

.. code-block:: console

   $ cmake --build .

To install heyoka, you can use the following CMake command:

.. code-block:: console

   $ cmake  --build . --target install

The installation command will copy the heyoka headers and library to the
``CMAKE_INSTALL_PREFIX`` directory.

If you enabled the ``HEYOKA_BUILD_TESTS`` option, you can run the test suite
with the following command:

.. code-block:: console

   $ cmake  --build . --target test

.. note::

   On Windows, and if heyoka is built as a shared library (the default),
   in order to execute the test or the benchmark suite you have to ensure that the
   ``PATH`` variable includes the directory that contains the heyoka
   DLL (otherwise the tests will fail to run).

Including heyoka in your project via CMake
------------------------------------------

As a part of the heyoka installation, a group of CMake files is installed into
``CMAKE_INSTALL_PREFIX/lib/cmake/heyoka``.
This bundle, which is known in the CMake lingo as a `config-file package <https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html>`__,
facilitates the detection and use of heyoka from other CMake-based projects.
heyoka's config-file package, once loaded, provides
an imported target called ``heyoka::heyoka`` which encapsulates all the information
necessary to use heyoka. That is, linking to
``heyoka::heyoka`` ensures that heyoka's include directories are added to the include
path of the compiler, and that the libraries
on which heyoka depends are brought into the link chain.

For instance, a ``CMakeLists.txt`` file for a project using heyoka
may look like this:

.. code-block:: cmake

   # The name of our project.
   project(sample_project)

   # Look for an installation of heyoka in the system.
   find_package(heyoka REQUIRED)

   # Create an executable, and link it to the heyoka::heyoka imported target.
   # This ensures that, in the compilation of 'main', heyoka's include
   # dirs are added to the include path of the compiler and that heyoka's
   # dependencies are transitively linked to 'main'.
   add_executable(main main.cpp)
   target_link_libraries(main heyoka::heyoka)
