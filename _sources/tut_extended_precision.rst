.. _tut_extended_precision:

Computations in extended precision
==================================

As hinted in the :ref:`installation instructions <ep_support>`, heyoka supports computations
not only in single and double precision, but also in extended precision. Specifically, heyoka currently supports:

* the 80-bit IEEE `extended-precision format <https://en.wikipedia.org/wiki/Extended_precision>`__ (~21 decimal digits),
* the 128-bit IEEE `quadruple-precision format <https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format>`__ (~36 decimal digits).

How these extended precision floating-point types can be accessed and used from C++ varies depending on the platform. The 80-bit
extended-precision format is available as the C++ ``long double`` type on most platforms based on Intel x86 processors. Quadruple-precision
computations are supported either via the ``long double`` type (e.g., on 64-bit Linux ARM) or via the the :cpp:class:`mppp::real128` type
(provided that the platform supports the nonstandard ``__float128`` floating-point type and that heyoka was compiled with support
for the mp++ library - see the :ref:`installation instructions <installation>`).

A simple example
----------------

We will be assuming here that ``long double`` implements the 80-bit extended-precision floating-point format.
In order to verify that heyoka indeed is able to work in extended precision, we will be monitoring the evolution of the energy constant
in a high-precision numerical integration of the simple pendulum.

Let us begin as usual with the definition of the dynamical equations and the creation of the integrator object:

.. literalinclude:: ../tutorial/extended_precision.cpp
   :language: c++
   :lines: 20-31

In order to activate extended precision, we created an integrator object of type ``taylor_adaptive<long double>`` - that is,
we specified ``long double``, instead of the usual ``double``, as the (only) template parameter for the ``taylor_adaptive`` class template.
Note that, for simplicity, we still used double-precision values for the initial state: these values are automatically
converted to ``long double`` by the integrator's constructor. Note also that, when operating in extended precision,
*all* numerical values encapsulated in an integrator are represented in extended precision - this includes not only the state vector,
but also the time coordinate, the tolerance, the Taylor coefficients, etc. Similarly to double-precision integrators, the default value
of the tolerance is the machine epsilon of ``long double``.

Next, we define a small helper function that will allow us to monitor the evolution of the energy constant
throughout the integration:

.. literalinclude:: ../tutorial/extended_precision.cpp
   :language: c++
   :lines: 33-39

Before starting the integration, we compute and store the initial energy for later use:

.. literalinclude:: ../tutorial/extended_precision.cpp
   :language: c++
   :lines: 41-42

We can now begin a step-by-step integration. At the end of each step, we will be computing
and printing to screen the relative energy error:

.. literalinclude:: ../tutorial/extended_precision.cpp
   :language: c++
   :lines: 44-51

.. code-block:: console

   Relative energy error: 0
   Relative energy error: 0
   Relative energy error: 9.62658e-20
   Relative energy error: 0
   Relative energy error: 9.62658e-20
   Relative energy error: 9.62658e-20
   Relative energy error: 9.62658e-20
   Relative energy error: 1.92532e-19
   Relative energy error: 1.92532e-19
   Relative energy error: 1.92532e-19
   Relative energy error: 1.92532e-19
   Relative energy error: 9.62658e-20
   Relative energy error: 1.92532e-19
   Relative energy error: 9.62658e-20
   Relative energy error: 9.62658e-20
   Relative energy error: 9.62658e-20
   Relative energy error: 9.62658e-20
   Relative energy error: 9.62658e-20
   Relative energy error: 0
   Relative energy error: 9.62658e-20

The console output indeed confirms that energy is conserved at the level of the epsilon of the 80-bit
extended-precision format (that is, :math:`\sim 10^{-19}`).

Performing quadruple-precision integrations via the :cpp:class:`mppp::real128` type follows exactly the same pattern. That is,
one just needs to replace ``long double`` with :cpp:class:`mppp::real128` when instantiating the integrator object.

Other classes and functions
---------------------------

Besides the adaptive integrator, several other classes and functions in heyoka can be used in extended precision.

The :ref:`event classes <tut_events>`, for instance, can be constructed in extended precision by passing ``long double``
or :cpp:class:`mppp::real128` as the template parameter (instead of ``double``). Note that the precision of an event
must match the precision of the integrator object in which the event is used, otherwise an error will be produced
at compilation time.

Full code listing
-----------------

.. literalinclude:: ../tutorial/extended_precision.cpp
   :language: c++
   :lines: 9-
