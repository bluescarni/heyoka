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
can be set and changed at runtime.

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
precision value which is inferred upon construction from the precision of the initial conditions. This global precision
value can be accessed after construction via the ``get_prec()`` member function. If the initial
state vector contains values with different precisions, or if any other numerical value passed to the constructor
has a precision inconsistent with the initial conditions, an exception will be thrown. Moreover, if at any time
the user changes the precision of the internal data of the integrator, any successive attempt to continue the numerical
integration will result in an exception being raised. 
In other words, heyoka will not accept any change to the precision of the numerical data stored within an
arbitrary-precision integrator.

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
being implemented in LLVM).

Extended vs arbitrary precision
-------------------------------

