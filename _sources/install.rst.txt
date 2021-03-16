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
* the `{fmt} <https://fmt.dev/latest/index.html>`__ library,
* the `spdlog <https://github.com/gabime/spdlog>`__ library.

Additionally, heyoka has the following **optional** dependencies:

* the `mp++ <https://bluescarni.github.io/mppp/>`__ multiprecision library
  (provides support for quadruple-precision integrations),
* the `SLEEF <https://sleef.org/>`__ vectorized math library (improves the performance
  of integrations in batch mode),
* the `xtensor and xtensor-blas <https://xtensor.readthedocs.io/en/latest/>`__
  libraries (used in the tests and benchmarks).

`CMake <https://cmake.org/>`__ is the build system used by heyoka and it must also be available when
installing from source (the minimum required version is 3.8).

.. _ep_support:

Support for extended precision
``````````````````````````````

Whereas in heyoka double-precision computations are always supported, support for extended-precision
computations varies depending on the software/hardware platform.

80-bit precision
^^^^^^^^^^^^^^^^

80-bit precision support requires an x86 processor. Additionally, the 80-bit floating-point
type must be available in C++ as the ``long double`` type. This is case for most compilers,
with the notable exception of Microsoft Visual C++, in which ``long double`` is a synonym for ``double``.
Thus, on Windows heyoka (and all its dependencies) must be compiled with a non-MSVC compiler
in order to enable 80-bit computations.

128-bit precision
^^^^^^^^^^^^^^^^^

Currently quadruple-precision support in heyoka has the following prerequisites:

* either GCC or Clang must be used,
* the nonstandard ``__float128`` floating-point type must be
  `available and supported <https://gcc.gnu.org/onlinedocs/gcc/Floating-Types.html>`__,
* heyoka must be compiled with the ``HEYOKA_WITH_MPPP`` option enabled (see :ref:`below <installation_from_source>`).

Packages
--------

Conda
`````

heyoka is available via the `conda <https://conda.io/docs/>`__ package manager for Linux, OSX and Windows
thanks to the infrastructure provided by `conda-forge <https://conda-forge.org/>`__.

In order to install heyoka via conda, you just need to add ``conda-forge``
to the channels, and then we can immediately install heyoka:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda config --set channel_priority strict
   $ conda install heyoka

The conda package for heyoka is maintained by the core development team,
and it is regularly updated when new heyoka versions are released.

Please refer to the `conda documentation <https://conda.io/docs/>`__ for instructions
on how to setup and manage
your conda installation.

.. _installation_from_source:

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
This bundle, which is known in the CMake lingo as a
`config-file package <https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html>`__,
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

   # heyoka requires at least CMake 3.8.
   cmake_minimum_required(VERSION 3.8.0)

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
