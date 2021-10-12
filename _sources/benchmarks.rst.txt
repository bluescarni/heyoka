Benchmarks
==========

In this section we provide a few performance comparisons between heyoka and other popular
ODE integration packages. Specifically, we compare heyoka to:

- `DifferentialEquations.jl <https://diffeq.sciml.ai/>`__, a popular Julia
  library implementing several ODE solvers. In these benchmarks, we will be using
  the ``Vern9`` solver (which is the `recommended solver <https://diffeq.sciml.ai/stable/solvers/ode_solve/>`__
  for high-precision integrations); 
- `TaylorIntegration.jl <https://github.com/PerezHz/TaylorIntegration.jl>`__, a
  Julia-based implementation of Taylor's integration method. In these benchmarks, we will ensure
  that the Taylor order and tolerances for TaylorIntegration.jl match those selected by heyoka;
- `Boost.ODEInt <https://www.boost.org/doc/libs/master/libs/numeric/odeint/doc/html/index.html>`__,
  a C++ package implementing various algorithms for the solution of systems of ODEs. In these
  benchmarks, we will be using the Runge-Kutta Fehlberg 78 solver;
- The ``IAS15`` integrator from `REBOUND <https://github.com/hannorein/rebound>`__,
  a popular N-body integration package. Like heyoka, ``IAS15`` is a high-precision
  non-symplectic integrator with adaptive timestepping capable of conserving the
  dynamical invariants over billions of dynamical timescales. Note, however, that
  ``IAS15`` is not a general-purpose integrator, and thus we will be able to use
  it only in benchmarks involving gravitational N-body systems.

The Taylor integrator from `TaylorIntegration.jl <https://github.com/PerezHz/TaylorIntegration.jl>`__
can be sped up via the use of the ``@taylorize`` macro, which however imposes some limitations
on the form of the right-hand sides of the ODEs
(more details `here <https://perezhz.github.io/TaylorIntegration.jl/latest/taylorize/>`__). In the benchmarks,
we will measure the runtime of TaylorIntegration.jl with and without the ``@taylorize`` macro.

All benchmarks were run on an Intel Xeon Platinum 8360Y CPU. More benchmark results are available in the
`heyoka paper <https://arxiv.org/abs/2105.00800>`__.

The benchmarks' source code is available in the `github repository <https://github.com/bluescarni/heyoka/tree/master/benchmark>`__.

The simple pendulum
===================

In this first benchmark, we numerically integrate the `simple pendulum <https://en.wikipedia.org/wiki/Pendulum_(mathematics)>`__

.. math::

   \begin{cases}
   x^\prime = v \\
   v^\prime = -\sin x
   \end{cases},

with initial conditions

.. math::

   \begin{cases}
   x\left( 0 \right) &= 1.3 \\
   v\left( 0 \right) &= 0
   \end{cases},

for :math:`10^4` time units. The tolerance for all integrators is set to :math:`10^{-15}`.

Here are the results:

.. image:: images/pendulum_bench.png
  :align: center
  :alt: Pendulum benchmark

We can see how heyoka is between 5 and 32 times faster than the other tested integrators.

The three-body problem
======================

Here we will numerically integrate the planar circular restricted `three-body problem <https://en.wikipedia.org/wiki/Three-body_problem>`__
(PCR3BP). The dynamical equations are:

.. math::

   \begin{cases}
    \dot{x} & = p_x + y\\
    \dot{y} & = p_y - x\\
    \dot{p_x} & = - \frac{(1-\mu)(x-\mu)}{((x-\mu)^2+y^2)^{3/2}} - \frac{\mu(x+1-\mu)}{((x+1-\mu)^2+y^2)^{3/2}} + p_y\\
    \dot{p_y} & = - \frac{(1-\mu)y      }{((x-\mu)^2+y^2)^{3/2}} - \frac{\mu y       }{((x+1-\mu)^2+y^2)^{3/2}} - p_x
   \end{cases},

with mass parameter :math:`\mu = 0.01` and initial conditions

.. math::

   \begin{cases}
   x\left( 0 \right) & = -0.8 \\
   y\left( 0 \right) & = 0 \\
   p_x\left( 0 \right) & = 0 \\
   p_y\left( 0 \right) & = -0.6276410653920694 \\
   \end{cases},

for :math:`2 \times 10^3` time units. The tolerance for all integrators is set to :math:`10^{-15}`.

Here are the results:

.. image:: images/pcr3bp_bench.png
  :align: center
  :alt: PCR3BP benchmark

We can see how heyoka is between 8 and 62 times faster than the other tested integrators.

The outer Solar System
======================

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
