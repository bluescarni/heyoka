.. heyoka documentation master file, created by
   sphinx-quickstart on Fri Dec 18 00:10:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

heyoka
======

    The `heyókȟa <https://en.wikipedia.org/wiki/Heyoka>`__ [...] is a kind of
    sacred clown in the culture of the Sioux (Lakota and Dakota people)
    of the Great Plains of North America. The heyoka is a contrarian, jester,
    and satirist, who speaks, moves and reacts in an opposite fashion to the
    people around them.

heyoka is a C++ library for the integration of ordinary differential equations
(ODEs) via Taylor's method. Notable features include:

* support for both double-precision and extended-precision floating-point types
  (80-bit and 128-bit),
* the ability to maintain machine precision accuracy over
  tens of billions of timesteps,
* high-precision zero-cost dense output,
* batch mode integration to harness the power of modern
  `SIMD <https://en.wikipedia.org/wiki/SIMD>`__ instruction sets,
* a high-performance implementation of Taylor's method based
  on automatic differentiation techniques and aggressive just-in-time
  compilation via `LLVM <https://llvm.org/>`__.

As a simple example, consider the ODE system
corresponding to the `pendulum <https://en.wikipedia.org/wiki/Pendulum_(mathematics)>`__,

.. math::

   \begin{cases}
   x^\prime = v \\
   v^\prime = -9.8 \sin x
   \end{cases}

with initial conditions

.. math::

   \begin{cases}
   x\left( 0 \right) = 0.05 \\
   v\left( 0 \right) = 0.025
   \end{cases}

Here's how the ODE system is defined and numerically integrated
in heyoka:

.. literalinclude:: ../tutorial/pendulum.cpp
   :language: c++
   :lines: 9-

heyoka is released under the `MPL-2.0 <https://www.mozilla.org/en-US/MPL/2.0/FAQ/>`__
license. The authors are Francesco Biscani (Max Planck Institute for Astronomy) and
Dario Izzo (European Space Agency).

If you prefer using Python rather than C++, heyoka can be used from Python via
`heyoka.py <https://github.com/bluescarni/heyoka.py>`__, its Python bindings.

.. toctree::
   :maxdepth: 2

   install.rst
   basic_tutorials.rst
   ad_notes.rst
   changelog.rst
   bibliography.rst
