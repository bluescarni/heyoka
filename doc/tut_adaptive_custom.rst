.. _tut_adaptive_custom:

Customising the adaptive integrator
===================================

In the :ref:`previous section <tut_adaptive>` we showed a few
usage examples of the ``taylor_adaptive`` class using the default
options. Here, we will show how the behaviour of the integrator
can be customised in a variety of ways.

Error tolerance
---------------

As we mentioned earlier, by default the ``taylor_adaptive`` class
uses an error tolerance :math:`\varepsilon` equal to the machine
epsilon of the floating-point type in use. E.g., when using the
``double`` floating-point type, the tolerance is set to
:math:`\sim 2.2\times 10^{-16}`.

The :math:`\varepsilon` value is used by the ``taylor_adaptive``
class to control the error arising from truncating the (infinite)
Taylor series representing the solution of the ODE system.
In other words, ``taylor_adaptive`` strives to ensure that the
magnitude of the remainders of the Taylor series is
not greater than :math:`\varepsilon`,
either in an absolute or relative sense. Absolute error control mode
is activated when all elements of the state vector have a magnitude
less than 1, while relative error control mode is activated when at least one
element of the state vector has a magnitude greater than 1.

In order to specify a non-default tolerance, the keyword argument
``eps`` can be used when constructing an integrator object:
