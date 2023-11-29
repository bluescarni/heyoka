.. _tut_single_precision:

Computations in single precision
================================

.. versionadded:: 3.2.0

In previous tutorials we saw how heyoka, in addition to the standard
`double precision <https://en.wikipedia.org/wiki/Double-precision_floating-point_format>`__,
also supports computations in :ref:`extended precision <tut_extended_precision>` and
:ref:`arbitrary precision <tut_arbitrary_precision>`. Starting with version 3.2.0, heyoka
supports also computations in `single precision <https://en.wikipedia.org/wiki/Single-precision_floating-point_format>`__.

Single-precision computations can lead to substantial performance benefits when high accuracy is not required.
In particular, single-precision :ref:`batch mode <tut_batch_mode>` can use a SIMD width twice larger
than double precision, leading to an increase by a factor of 2 of the computational throughput.
In scalar computations, the use of single precision reduces by half the memory usage with respect to double precision,
which can help alleviating performance issues in large ODE systems. This can be particularly noticeable in applications such as
:external:ref:`neural ODEs <tut_neural_ode>`.

In C++, single-precision values are usually represented via the standard floating-point type ``float``.
Correspondingly, and similarly to what explained in the :ref:`extended precision <tut_extended_precision>`
tutorial, single-precision computations are activated by passing the ``float`` template parameter to functions
and classes in the heyoka API.

A simple example
----------------

In order to verify that heyoka indeed is able to work in single precision, we will be monitoring the evolution of the energy constant
in a low-precision numerical integration of the simple pendulum.

Let us begin as usual with the definition of the dynamical equations and the creation of the integrator object:

.. literalinclude:: ../tutorial/single_precision.cpp
   :language: c++
   :lines: 18-29

In order to activate single precision, we created an integrator object of type ``taylor_adaptive<float>`` - that is,
we specified ``float``, instead of the usual ``double``, as the (only) template parameter for the ``taylor_adaptive`` class template.
Note that we specified a single-precision initial state via the use of the ``f`` suffix for the numerical constants.
Note also that, when operating in single precision,
*all* numerical values encapsulated in an integrator are represented in single precision - this includes not only the state vector,
but also the time coordinate, the tolerance, the Taylor coefficients, etc. Similarly to double-precision integrators, the default value
of the tolerance is the machine epsilon of ``float``.

Next, we define a small helper function that will allow us to monitor the evolution of the energy constant
throughout the integration:

.. literalinclude:: ../tutorial/single_precision.cpp
   :language: c++
   :lines: 31-37

Before starting the integration, we compute and store the initial energy for later use:

.. literalinclude:: ../tutorial/single_precision.cpp
   :language: c++
   :lines: 39-40

We can now begin a step-by-step integration. At the end of each step, we will be computing
and printing to screen the relative energy error:

.. literalinclude:: ../tutorial/single_precision.cpp
   :language: c++
   :lines: 42-49

.. code-block:: console

   Relative energy error: 1.48183e-07
   Relative energy error: 5.29227e-08
   Relative energy error: 6.08611e-08
   Relative energy error: 1.79937e-07
   Relative energy error: 1.74645e-07
   Relative energy error: 2.24921e-07
   Relative energy error: 2.4609e-07
   Relative energy error: 1.1643e-07
   Relative energy error: 1.79937e-07
   Relative energy error: 1.40245e-07
   Relative energy error: 2.54029e-07
   Relative energy error: 1.84899e-07
   Relative energy error: 1.83245e-07
   Relative energy error: 1.56122e-07
   Relative energy error: 2.22275e-07
   Relative energy error: 1.61414e-07
   Relative energy error: 2.11691e-07
   Relative energy error: 2.88428e-07
   Relative energy error: 2.93721e-07
   Relative energy error: 1.82583e-07

The console output indeed confirms that energy is conserved at the level of the epsilon of the
single-precision format (that is, :math:`\sim 10^{-7}`).

Other classes and functions
---------------------------

Besides the adaptive integrator, several other classes and functions in heyoka can be used in single precision.

The :ref:`event classes <tut_events>`, for instance, can be constructed in single precision by passing ``float``
as the template parameter (instead of ``double``). Note that the precision of an event
must match the precision of the integrator object in which the event is used, otherwise an error will be produced
at compilation time.

Full code listing
-----------------

.. literalinclude:: ../tutorial/single_precision.cpp
   :language: c++
   :lines: 9-
