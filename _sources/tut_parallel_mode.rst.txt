.. _tut_parallel_mode:

Parallel mode
=============

.. versionadded:: 0.18.0

Starting from version 0.18.0, heyoka can automatically parallelise
the integration of a single ODE system using multiple threads
of execution. This parallelisation
mode is fine-grained, i.e., it acts at the level of an individual
integration step, and it thus serves a fundamentally different purpose from
the coarse-grained parallelisation approach of :ref:`ensemble propagations <tut_ensemble>`.

In order to be effective, parallel mode needs large ODE systems, that is, systems
with a large number of variables and/or large expressions at the right-hand side.
When used on small ODE systems, parallel mode will likely introduce a noticeable
slowdown due to the multithreading overhead.

Note that, because Taylor integrators are memory intensive, performance
for large ODE systems is bottlenecked by RAM speed due to the
`memory wall <https://en.wikipedia.org/wiki/Random-access_memory#Memory_wall>`__.
This means, in practice, that, at least for double-precision computations,
the performance of parallel mode will not scale linearly with the number of cores.
On the other hand, for extended-precision computations the speedup will be much more efficient,
due to the fact that arithmetic operations on extended-precision operands
are computationally heavier than on double-precision operands.

With these caveats out of the way, let us see an example of parallel mode in action.

Parallel planetary embryos
--------------------------

In order to illustrate the effectiveness of parallel mode, we will setup
an N-body system consisting of a large number (:math:`N=400`) of
`protoplanets <https://en.wikipedia.org/wiki/Protoplanet>`__ in orbit around
a Sun-like star. The protoplanets interact with the star and with each other according to
Newtonian gravity, and they are initially placed on circular orbits.

Let us begin by defining a ``run_benchmark()`` function that will setup and integrate
the N-body system. The function is parametrised over the floating-point type ``T`` that
will be used for the integration (so that we can easily run the same test in both double
and quadruple precision). The input arguments are the final time and a boolean flag
specifying whether or not to use parallel mode. The return value is the total wall clock time
taken by the integration:

.. literalinclude:: ../tutorial/par_mode.cpp
   :language: c++
   :lines: 32-85

Parallel mode can be enabled when constructing the integrator object:

.. literalinclude:: ../tutorial/par_mode.cpp
   :language: c++
   :lines: 56-59

Note that parallel mode **requires** compact mode: if you try to construct a parallel mode integrator
without enabling compact mode, an exception will be thrown.

Before running the benchmarks, we will limit the number of threads available for use by heyoka to 8. Since heyoka
uses internally the `TBB <https://github.com/oneapi-src/oneTBB>`__ library for multithreading, we will
use the TBB API to achieve this goal:

.. literalinclude:: ../tutorial/par_mode.cpp
   :language: c++
   :lines: 87-90

Let us now run the benchmark in double precision, and let us compare the timings with and without parallel mode:

.. literalinclude:: ../tutorial/par_mode.cpp
   :language: c++
   :lines: 92-97

.. code-block:: console

   Serial time (double): 21107ms
   Parallel time (double): 5887ms

We can see how parallel mode resulted in a :math:`\times 3.6` speedup with respect to the serial integration.
While this speedup is suboptimal with respect to the maximum theoretically achievable speedup of :math:`\times 8`,
these timings show that parallel mode can still provide an easy way of boosting the integrator's performance
for large ODE systems.

Let us now repeat the same test in quadruple precision:

.. literalinclude:: ../tutorial/par_mode.cpp
   :language: c++
   :lines: 101-106

.. code-block:: console

   Serial time (real128): 210398ms
   Parallel time (real128): 29392ms

In quadruple precision, the speedup is now :math:`\times 7.2`, which is quite close to optimal.

These results show that parallel mode can provide an easy way of boosting the performance of heyoka's integrators
for large ODE systems. For double-precision computations the speedup is suboptimal due to the high memory
usage of Taylor integrators. For quadruple-precision computations, the speedup is close to optimal.

Full code listing
-----------------

.. literalinclude:: ../tutorial/par_mode.cpp
   :language: c++
   :lines: 9-
