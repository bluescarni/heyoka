.. _benchmarks:

Benchmarks
==========

In this section we provide a few performance comparisons between heyoka and other popular
ODE integration packages. Specifically, we compare heyoka to:

- `DifferentialEquations.jl <https://diffeq.sciml.ai/>`__, a popular Julia
  library implementing several ODE solvers. In these benchmarks, we will be using
  the ``Vern9`` explicit Runge-Kutta solver (which is the
  `recommended solver <https://diffeq.sciml.ai/stable/solvers/ode_solve/>`__
  for low-tolerance integrations of non-stiff problems in double precision);
- `Boost.ODEInt <https://www.boost.org/doc/libs/master/libs/numeric/odeint/doc/html/index.html>`__,
  a C++ package implementing various algorithms for the solution of systems of ODEs. In these
  benchmarks, we will be using the explicit Runge-Kutta-Fehlberg 78 solver;
- The ``IAS15`` integrator from `REBOUND <https://github.com/hannorein/rebound>`__,
  a popular N-body integration package. Like heyoka, ``IAS15`` is a high-precision
  non-symplectic integrator with adaptive timestepping capable of conserving the
  dynamical invariants over billions of dynamical timescales. Note, however, that
  ``IAS15`` is not a general-purpose integrator, and thus we will be able to use
  it only in benchmarks involving gravitational N-body systems.

Note that the ``Vern9`` integrator from `DifferentialEquations.jl <https://diffeq.sciml.ai/>`__ by default
enables dense output, which however incurs in a heavy computational cost. While heyoka also supports
:ref:`dense output <tut_d_output>`, this feature is opt-in and its performance impact is much more limited.
In these benchmarks, we will be testing both heyoka and ``Vern9`` with and without dense output.

All benchmarks were run on an Intel Xeon Platinum 8360Y CPU. More benchmark results are available in the
`heyoka paper <https://arxiv.org/abs/2105.00800>`__.
The benchmarks' source code is available in the `github repository <https://github.com/bluescarni/heyoka/tree/master/benchmark>`__.

The planetary three-body problem
--------------------------------

Here we will numerically integrate a specific case of the `three-body problem <https://en.wikipedia.org/wiki/Three-body_problem>`__
in which the three particles are the Sun, Jupiter and Saturn, all represented as point masses
attracting each other according to `Newtonian gravity <https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation>`__.
The initial conditions are taken from `this paper <https://ntrs.nasa.gov/citations/19860060859>`__, and the integration
is run for a total of :math:`10^5` years.
For ``Vern9`` and heyoka, the test is run both with and without :ref:`dense output <tut_d_output>`. When dense output is enabled,
the result of the integration over :math:`5 \times 10^5` equispaced time grid points is requested.

In order to measure the accuracy of the integration, we will also compare the final state of the system
with the result of a numerical integration in quadruple precision with a tolerance of :math:`10^{-30}`.

Let us see first the results for an error tolerance of :math:`10^{-15}`:

.. image:: images/ss_3bp_15.png
  :align: center
  :alt: 3BP benchmark 1e-15

We can see how, without dense output, heyoka is about 3 times faster than ``Vern9``. When dense output is requested,
heyoka's runtime increases by a modest :math:`\sim 24\%`, whereas for ``Vern9`` the runtime increases by a factor of
:math:`\sim 3`, so that, with dense output, heyoka is about :math:`\sim 7` times faster than ``Vern9``. Performance-wise,
Boost.ODEint is comparable to ``Vern9`` (note that Boost.ODEInt does not support dense output).

From the point of view of the integration accuracy, we can see how the RMS of the error across the components of the state
vector with respect to the quadruple-precision integration is of the order of :math:`10^{-9}` for heyoka, while for both
``Vern9`` and Boost.ODEInt the error is about :math:`\sim 35` times larger.

Note that, even if the error tolerance for the integration is set to :math:`10^{-15}`, the error at the end of the integration
is of the order of :math:`10^{-9}`. This is to be expected, as the error on the state variables accumulates as (at least)
:math:`t^{\frac{3}{2}}` (a result known as Brouwer's law).

Let us now see the results for an error tolerance of :math:`10^{-9}`:

.. image:: images/ss_3bp_9.png
  :align: center
  :alt: 3BP benchmark 1e-9

Whereas heyoka is still faster than ``Vern9`` and Boost.ODEInt, at this higher integration tolerance the performance
advantage is smaller.

The integration accuracy of both heyoka and ``Vern9`` is of the order of :math:`10^{-3}`. By contrast,
the accuracy of Boost.ODEInt is two orders of magnitude worse.

The outer Solar System
----------------------

In this benchmark, we will integrate the motion of the outer Solar System for 1 million years. We define the outer Solar
System as the 6-body problem consisting of the Sun, Jupiter, Saturn, Uranus, Neptune and Pluto, all considered as point
masses attracting each other according to `Newtonian gravity <https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation>`__.
The initial conditions are taken from `this paper <https://ntrs.nasa.gov/citations/19860060859>`__.

For this benchmark, we will be comparing heyoka to the ``IAS15`` integrator from `REBOUND <https://github.com/hannorein/rebound>`__.

Here are the results:

.. image:: images/outer_ss_bench.png
  :scale: 60%
  :align: center
  :alt: Outer Solar System benchmark

We can see how heyoka is about 5 times faster than ``IAS15`` in this specific test.
