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
