heyoka
======

[![Build Status](https://img.shields.io/circleci/project/github/bluescarni/heyoka/master.svg?style=for-the-badge)](https://circleci.com/gh/bluescarni/heyoka)
[![Build Status](https://img.shields.io/appveyor/ci/bluescarni/heyoka/master.svg?logo=appveyor&style=for-the-badge)](https://ci.appveyor.com/project/bluescarni/heyoka)
[![Build Status](https://img.shields.io/github/workflow/status/bluescarni/heyoka/GitHub%20CI?style=for-the-badge)](https://github.com/bluescarni/heyoka/actions?query=workflow%3A%22GitHub+CI%22)
![language](https://img.shields.io/badge/language-C%2B%2B17-red.svg?style=for-the-badge)

[![Anaconda-Server Badge](https://img.shields.io/conda/vn/conda-forge/heyoka.svg?style=for-the-badge)](https://anaconda.org/conda-forge/heyoka)

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/bluescarni/heyoka">
    <img src="doc/images/white_logo.png" alt="Logo" width="280">
  </a>
  <p align="center">
    Modern Taylor's method via just-in-time compilation
    <br />
    <a href="https://bluescarni.github.io/heyoka/index.html"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/bluescarni/heyoka/issues/new/choose">Report bug</a>
    ·
    <a href="https://github.com/bluescarni/heyoka/issues/new/choose">Request feature</a>
  </p>
</p>


> The [heyókȟa](https://en.wikipedia.org/wiki/Heyoka>) [...] is a kind of
> sacred clown in the culture of the Sioux (Lakota and Dakota people)
> of the Great Plains of North America. The heyoka is a contrarian, jester,
> and satirist, who speaks, moves and reacts in an opposite fashion to the
> people around them.

heyoka is a C++ library for the integration of ordinary differential equations
(ODEs) via Taylor's method. Notable features include:

* support for both double-precision and extended-precision floating-point types
  (80-bit and 128-bit),
* the ability to maintain machine precision accuracy over
  tens of billions of timesteps,
* high-precision zero-cost dense output,
* batch mode integration to harness the power of modern
  [SIMD](https://en.wikipedia.org/wiki/SIMD) instruction sets,
* a high-performance implementation of Taylor's method based
  on automatic differentiation techniques and aggressive just-in-time
  compilation via [LLVM](https://llvm.org/).

Quick example
-------------

As a simple example, here's how the ODE system of the
[pendulum](https://en.wikipedia.org/wiki/Pendulum_(mathematics))
is defined and numerically integrated
in heyoka:

```c++
#include <iostream>

#include <heyoka/heyoka.hpp>

using namespace heyoka;

int main()
{
    // Create the symbolic variables x and v.
    auto [x, v] = make_vars("x", "v");

    // Create the integrator object
    // in double precision.
    auto ta = taylor_adaptive<double>{// Definition of the ODE system:
                                      // x' = v
                                      // v' = -9.8 * sin(x)
                                      {prime(x) = v, prime(v) = -9.8 * sin(x)},
                                      // Initial conditions
                                      // for x and v.
                                      {0.05, 0.025}};

    // Integrate for 10 time units.
    ta.propagate_for(10.);

    // Print the state vector.
    std::cout << "x(10) = " << ta.get_state()[0] << '\n';
    std::cout << "v(10) = " << ta.get_state()[1] << '\n';
}
```

Documentation
-------------

The full documentation can be found [here](https://bluescarni.github.io/heyoka/).

Authors
-------

* Francesco Biscani (Max Planck Institute for Astronomy)
* Dario Izzo (European Space Agency)

License
-------

heyoka is released under the [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/FAQ/)
license.
