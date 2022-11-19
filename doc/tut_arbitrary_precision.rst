.. _tut_arbitrary_precision:

Computations in arbitrary precision
===================================

.. versionadded:: 0.20.0

In addition to :ref:`extended-precision computations <tut_extended_precision>`, heyoka also supports
computations in *arbitrary* precision. In arbitrary-precision computations the user can
decide at runtime the number of bits of precision to be employed in the representation
of floating-point numbers. The precision is limited, in principle, only by the available
memory.

As explained in the :ref:`installation instructions <installation>`, arbitrary-precision mode
requires heyoka to be compiled with support for the `mp++ <https://bluescarni.github.io/mppp/>`__
multiprecision library. The mp++ library, in turn, must be compiled
with the ``MPPP_WITH_MPFR`` option enabled (see the :ref:`mp++ installation instructions <mppp:installation>`).

Arbitrary-precision floating-point values are represented in heyoka via the mp++ :cpp:class:`mppp::real` class.
We refer the reader to the :ref:`mp++ tutorial <mppp:tutorial_real>` for a quick overview of the main features
of this class. For the purposes of this tutorial, it is sufficient to note how :cpp:class:`mppp::real` largely
behaves like a builtin floating-point type, the only difference being that the number of digits in the significand
can be set at runtime.

A simple example
----------------

In this tutorial, we will repeat the simple pendulum integration from the
:ref:`extended-precision tutorial <tut_extended_precision>`, but this time
in `octuple precision <https://en.wikipedia.org/wiki/Octuple-precision_floating-point_format>`__.
The number of binary digits in this representation is 237, corresponding to circa 71 decimal digits.

Let us begin as usual with the definition of the dynamical equations and the creation of the integrator object:

.. literalinclude:: ../tutorial/arbitrary_precision.cpp
   :language: c++
   :lines: 17-31

In order to activate arbitrary precision, we created an integrator object of type ``taylor_adaptive<mppp::real>`` - that is,
we specified :cpp:class:`mppp::real`, instead of the usual ``double``, as the (only) template parameter for the ``taylor_adaptive``
class template. The initial conditions were also initialised as a vector of :cpp:class:`mppp::real`, and each value in the initial
state vector was created with a precision of 237 bits. The precision of an :cpp:class:`mppp::real` is represented in mp++ via the 
:cpp:type:`mpfr_prec_t` type (an alias for a signed integral type).

Like in :ref:`extended-precision mode <tut_extended_precision>`, in an arbitrary-precision integrator
*all* numerical values encapsulated in an integrator are represented in arbitrary precision - this includes not only the state vector,
but also the time coordinate, the tolerance, the Taylor coefficients, etc. Each arbitrary-precision integrator has a global
precision value which is specified or inferred upon construction. This global precision
value can be accessed after construction via the ``get_prec()`` member function.

The precision value for an integrator can either be inferred from the initial state vector, as we saw in the
code snippet above, or it can be explicitly passed as the ``prec`` keyword argument. That is, an alternative,
but equivalent, way of constructing a 237-bit integrator would be:

.. code-block:: c++

    auto ta = taylor_adaptive<mppp::real>{// Definition of the ODE system:
                                          // x' = v
                                          // v' = -9.8 * sin(x)
                                          {prime(x) = v, prime(v) = -9.8 * sin(x)},
                                          // Initial conditions
                                          // for x and v.
                                          {1., 0.},
                                          // Explicitly set the precision.
                                          kw::prec = 237
                                          };

Regardless of how the integrator's precision is specified, all the numerical values encapsulated in the
integrator in addition to the state vector (that is, the time coordinate, the parameter values, the Taylor coefficients, etc.) are initialised
with the same precision upon construction. If at any time after construction
the user changes the precision of the internal data of the integrator, any successive attempt to continue the numerical
integration will result in an exception being raised. 
In other words, heyoka will not accept any change to the precision of the numerical data stored within an
arbitrary-precision integrator after construction.

Let us print to screen the integrator object:

.. literalinclude:: ../tutorial/arbitrary_precision.cpp
   :language: c++
   :lines: 33-34

.. code-block:: console

   Precision               : 237 bits
   Tolerance               : 9.055679078826712367509119290887791780682531198139138189582614889935501319e-72
   High accuracy           : false
   Compact mode            : true
   Taylor order            : 83
   Dimension               : 2
   Time                    : 0.000000000000000000000000000000000000000000000000000000000000000000000000
   State                   : [-1.000000000000000000000000000000000000000000000000000000000000000000000000, 0.000000000000000000000000000000000000000000000000000000000000000000000000]

The screen output indeed confirms that the precision was correctly inferred to be 237 bits from the initial conditions.
Similarly to double-precision and extended-precision integrators, the tolerance is set by default to the machine epsilon
corresponding to the inferred precision.

An important change in the defaults with respect to double-precision and extended-precision integrators is that, in
arbitrary-precision integrators, :ref:`compact mode <tut_compact_mode>` is **on** by default, rather than off.
This change is motivated by the fact that the Taylor order of arbitrary-precision integrators will typically be
much higher than in double-precision and extended-precision integrators, which results in very long compilation
times even for simple ODEs. Moreover, in arbitrary-precision integrators compact mode does not bring performance
improvements due to the fact that most numerical computations are offloaded to the mp++ library (rather than
being implemented directly in LLVM).

We proceed now to the definition of a small helper function that will allow us to monitor the evolution of the energy constant
throughout the integration. As we mentioned earlier, the :cpp:class:`mppp::real` class can be used like a builtin floating-point type, thus we can
copy without changes and re-use the generic ``compute_energy`` helper function from the
:ref:`extended-precision tutorial <tut_extended_precision>`:

.. literalinclude:: ../tutorial/arbitrary_precision.cpp
   :language: c++
   :lines: 36-42

Before starting the integration, we compute and store the initial energy for later use:

.. literalinclude:: ../tutorial/arbitrary_precision.cpp
   :language: c++
   :lines: 44-45

We can now begin a step-by-step integration. At the end of each step, we will be computing
and printing to screen the relative energy error:

.. literalinclude:: ../tutorial/arbitrary_precision.cpp
   :language: c++
   :lines: 47-54

.. code-block:: console

   Relative energy error: 8.04049e-72
   Relative energy error: 0
   Relative energy error: 0
   Relative energy error: 0
   Relative energy error: 8.04049e-72
   Relative energy error: 1.6081e-71
   Relative energy error: 8.04049e-72
   Relative energy error: 8.04049e-72
   Relative energy error: 1.6081e-71
   Relative energy error: 1.6081e-71
   Relative energy error: 1.6081e-71
   Relative energy error: 1.6081e-71
   Relative energy error: 8.04049e-72
   Relative energy error: 0
   Relative energy error: 8.04049e-72
   Relative energy error: 8.04049e-72
   Relative energy error: 8.04049e-72
   Relative energy error: 1.6081e-71
   Relative energy error: 8.04049e-72
   Relative energy error: 1.6081e-71

The console output indeed confirms that energy is conserved at the level of the epsilon of the octuple-precision
format (that is, :math:`\sim 10^{-71}`).

Other classes and functions
---------------------------

Besides the adaptive integrator, several other classes and functions in heyoka can be used in arbitrary precision.

The :ref:`event classes <tut_events>`, for instance, can be constructed in arbitrary precision by passing :cpp:class:`mppp::real`
as the template parameter (instead of ``double``). Note that arbitrary-precision events **must** be used in an arbitrary-precision
integrator, otherwise an error will be produced at compilation time.

Extended vs arbitrary precision
-------------------------------

In general, if your architecture supports :ref:`extended-precision data types <tut_extended_precision>` and their precision
is sufficient, you should prefer extended-precision integrations over arbitrary-precision integrations, as the former
will provide better performance.

On the other hand, arbitrary-precision integrations can satisfy any precision requirement, and they are available on all
platforms, even where there are no extended-precision data types (e.g., on Windows).
