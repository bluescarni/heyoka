.. _installation:

Installation
============

Introduction
------------

heyoka is written in modern C++, and it requires a compiler able to understand
at least C++17. The library is regularly tested on
a continuous integration pipeline which currently includes:

* GCC 9 on Linux,
* Clang 11 on OSX,
* MSVC 2019 and Clang 12 on Windows.

.. note::

   When using MSVC, heyoka currently requires MSVC>=2019. It is also possible
   to compile heyoka using the standard library from MSVC 2017 in conjunction
   with the ``clang-cl`` compiler.

The tested and supported CPU architectures at this time are x86-64,
64-bit ARM and 64-bit PowerPC.

heyoka has the following **mandatory** dependencies:

* the `LLVM <https://llvm.org/>`__ compiler infrastructure library (version >= 11),
* the `Boost <https://www.boost.org/>`__ C++ libraries (version >= 1.69),
* the `{fmt} <https://fmt.dev/latest/index.html>`__ library (version >= 9),
* the `spdlog <https://github.com/gabime/spdlog>`__ library,
* the `TBB <https://github.com/oneapi-src/oneTBB>`__ library.

Additionally, heyoka has the following **optional** dependencies:

* the `mp++ <https://bluescarni.github.io/mppp/>`__ multiprecision library,
  which provides support for arbitrary-precision integrations on all platforms,
  and for quadruple-precision integrations on platforms
  supporting the non-standard ``__float128`` type. heyoka requires
  an mp++ installation with support for Boost.serialization and for the
  {fmt} library
  (see the :ref:`mp++ installation instructions <mppp:installation>`).
  The minimum required version of mp++ is 0.27;
* the `SLEEF <https://sleef.org/>`__ vectorized math library (improves the performance
  of integrations in batch mode),
* the `xtensor and xtensor-blas <https://xtensor.readthedocs.io/en/latest/>`__
  libraries (used in the tests and benchmarks).

`CMake <https://cmake.org/>`__ is the build system used by heyoka and it must also be available when
installing from source (the minimum required version is 3.18).

.. warning::

   The `spdlog <https://github.com/gabime/spdlog>`__ library depends on the `{fmt} <https://fmt.dev/latest/index.html>`__ library,
   and by default spdlog uses a bundled internal copy of {fmt} which may not be compatible with other {fmt} installations
   that may be present on the system. This situation can lead to build and/or runtime errors.

   Users are thus advised to ensure that spdlog is built with the
   ``SPDLOG_FMT_EXTERNAL`` CMake option turned ``ON``, in order to ensure that both spdlog and heyoka are linking
   to the same {fmt} installation.

.. _ep_support:

Support for extended precision
``````````````````````````````

Whereas in heyoka single-precision and double-precision computations are always supported via the
``float`` and ``double`` types respectively, support for extended-precision
computations varies depending on the software/hardware platform.

80-bit precision
^^^^^^^^^^^^^^^^

80-bit precision support requires an x86 processor. Additionally, the 80-bit floating-point
type must be available in C++ as the ``long double`` type. This is case for most compilers,
with the notable exception of Microsoft Visual C++ (MSVC), where ``long double`` is a synonym for ``double``.
Thus, on Windows 80-bit precision support is **not** available, unless
heyoka (and all its dependencies) have been compiled with a compiler supporting the
80-bit floating-point type.

128-bit precision
^^^^^^^^^^^^^^^^^

On platforms where ``long double`` is a quadruple-precision floating-point datatype (e.g., 64-bit ARM),
quadruple-precision integrations are always supported via ``long double``. Otherwise,
on platforms such as x86-64, quadruple-precision computations are supported if:

* the nonstandard ``__float128`` floating-point type is
  available and supported, and
* an installation of the `mp++ <https://bluescarni.github.io/mppp/>`__ library with support
  for the :cpp:class:`mppp::real128` class is available (see the :ref:`mp++ installation instructions <mppp:installation>`),
  and
* heyoka is compiled with the ``HEYOKA_WITH_MPPP`` option enabled (see :ref:`below <installation_from_source>`).

If these conditions are satisfied, then quadruple-precision computations are supported in heyoka
via the :cpp:class:`mppp::real128` type.

.. note::

   The non-IEEE ``long double`` type available on some PowerPC platforms
   (which implements a double-length floating-point representation with 106
   significant bits) is **not** supported by heyoka at this time.

Arbitrary-precision
^^^^^^^^^^^^^^^^^^^

Arbitrary-precision integrations are supported on all platforms, provided that heyoka
is compiled with the ``HEYOKA_WITH_MPPP`` option enabled (see :ref:`below <installation_from_source>`)
and that the mp++ library is compiled with the ``MPPP_WITH_MPFR`` option enabled
(see the :ref:`mp++ installation instructions <mppp:installation>`).

Packages
--------

Conda
`````

heyoka is available via the `conda <https://docs.conda.io/en/latest/>`__ package manager for Linux, OSX and Windows
thanks to the infrastructure provided by `conda-forge <https://conda-forge.org/>`__.

In order to install heyoka via conda, you just need to add ``conda-forge``
to the channels, and then we can immediately install heyoka:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda config --set channel_priority strict
   $ conda install heyoka

Note that the ``heyoka`` package on conda is built against an unspecified version of LLVM. If you need
a package built against a *specific* version of LLVM, you can install one of the ``heyoka-llvm-*``
meta-packages. For instance, in order to install a package built against LLVM 12, you
could use the following command:

.. code-block:: console

   $ conda install heyoka-llvm-12

The list of heyoka meta-packages is available
`here <https://github.com/conda-forge/heyoka-feedstock>`__.

The conda packages for heyoka are maintained by the core development team,
and they are regularly updated when new heyoka versions are released.

Please refer to the `conda documentation <https://docs.conda.io/en/latest/>`__ for instructions
on how to setup and manage
your conda installation.

FreeBSD
```````

A community-supported FreeBSD port via `pkg <https://docs.freebsd.org/en/books/handbook/ports/#pkgng-intro>`__ is available for
heyoka. In order to install heyoka using pkg, execute the following command:

.. code-block:: console

   $ pkg install heyoka

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
  the heyoka library (requires compiler support, off by default).

The following advanced options are also available:

* ``HEYOKA_FORCE_STATIC_LLVM``: force statically linking to the LLVM libraries
  (off by default). Note that, by default, heyoka prefers to dynamically link to LLVM
  if both dynamic and static versions of the libraries are available.
* ``HEYOKA_HIDE_LLVM_SYMBOLS``: when statically linking to the LLVM libraries,
  try to hide the symbols exported by LLVM (off by default). When linking dynamically
  to LLVM, this option has no effects.

The ``HEYOKA_HIDE_LLVM_SYMBOLS`` option is useful if heyoka needs to be used in conjunction
with software linking to an LLVM version different from the one used by heyoka. In such
cases, symbol collisions between different LLVM version coexisting in the same process
will lead to unpredictable runtime behaviour (e.g., segfaults). This option attempts
to hide the LLVM symbols exported by the LLVM version in use by heyoka in order to
avoid symbol collisions. Note however that, depending on the platform, the
``HEYOKA_HIDE_LLVM_SYMBOLS`` option might end up hiding the symbols exported by
**all** the static libraries heyoka links to (i.e., not only LLVM),
which might end up creating other issues. Users are thus advised to activate this option
only if LLVM is the **only** static library heyoka links to.

In order to build heyoka, you can run the following CMake command from the
build directory:

.. code-block:: console

   $ cmake --build .

.. note::

   heyoka relies on a conforming implementation of IEEE floating-point
   arithmetic. Do *not* enable fast math flags (e.g., ``-ffast-math``,
   ``-Ofast``, etc.) when compiling heyoka or software depending on heyoka.
   If you are using the Intel C++ compiler, make sure that you are using
   the ``strict`` floating-point model.

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

   # heyoka requires at least CMake 3.18.
   cmake_minimum_required(VERSION 3.18.0)

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

heyoka's config-file package also exports the following boolean variables to signal with which optional
dependencies heyoka was compiled:

* ``heyoka_WITH_SLEEF`` if SLEEF support was enabled,
* ``heyoka_WITH_MPPP`` if mp++ support was enabled,
* ``heyoka_WITH_REAL128`` (new in version 0.19) if quadruple-precision
  computations via the :cpp:class:`mppp::real128` type are supported,
* ``heyoka_WITH_REAL`` (new in version 0.20) if arbitrary-precision
  computations via the :cpp:class:`mppp::real` type are supported.

.. versionadded:: 0.17.0

heyoka's config-file package also exports a
``heyoka_LLVM_VERSION_MAJOR`` variable containing
the major number of the LLVM version against which heyoka
was compiled. E.g., if heyoka was compiled against LLVM 12.0.1,
then ``heyoka_LLVM_VERSION_MAJOR`` is ``12``.
