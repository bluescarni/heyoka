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
* accurate and reliable event detection,
* batch mode integration to harness the power of modern
  `SIMD <https://en.wikipedia.org/wiki/SIMD>`__ instruction sets,
* a high-performance implementation of Taylor's method based
  on automatic differentiation techniques and aggressive just-in-time
  compilation via `LLVM <https://llvm.org/>`__.

If you prefer using Python rather than C++, heyoka can be used from Python via
`heyoka.py <https://github.com/bluescarni/heyoka.py>`__, its Python bindings.

If you are using heyoka as part of your research, teaching, or other activities, we would be grateful if you could star
the repository and/or cite our work. For citation purposes, you can use the following BibTex entry, which refers
to the heyoka paper (`arXiv preprint <https://arxiv.org/abs/2105.00800>`__):

.. code-block:: bibtex

   @article{10.1093/mnras/stab1032,
       author = {Biscani, Francesco and Izzo, Dario},
       title = "{Revisiting high-order Taylor methods for astrodynamics and celestial mechanics}",
       journal = {Monthly Notices of the Royal Astronomical Society},
       volume = {504},
       number = {2},
       pages = {2614-2628},
       year = {2021},
       month = {04},
       issn = {0035-8711},
       doi = {10.1093/mnras/stab1032},
       url = {https://doi.org/10.1093/mnras/stab1032},
       eprint = {https://academic.oup.com/mnras/article-pdf/504/2/2614/37750349/stab1032.pdf}
   }

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

.. toctree::
   :maxdepth: 2

   install.rst
   basic_tutorials.rst
   advanced_tutorials.rst
   ad_notes.rst
   changelog.rst
   breaking_changes.rst
   bibliography.rst
