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
``tol`` can be used when constructing an integrator object:

.. literalinclude:: ../tutorial/adaptive_opt.cpp
   :language: c++
   :lines: 17-34

.. code-block:: console

   Taylor order: 12
   Dimension   : 2
   Time        : 0.0000000000000000
   State       : [0.050000000000000003, 0.025000000000000001]

As you can see, the optimal Taylor order for a tolerance of :math:`10^{-9}`
is 12 (compared to a Taylor order of 20 for a tolerance
:math:`\sim 2.2\times 10^{-16}`).

Integrating the system back and forth shows how the accuracy of the
integration is decreased with respect to the default tolerance value:

.. literalinclude:: ../tutorial/adaptive_opt.cpp
   :language: c++
   :lines: 36-40

.. code-block:: console

   Taylor order: 12
   Dimension   : 2
   Time        : 0.0000000000000000
   State       : [0.050000000001312876, 0.024999999997558964]

Compact mode
------------

By default, the just-in-time compilation process of heyoka
aims at maximising runtime performance over everything else.
In practice, this means that heyoka generates a timestepper
function in which there are no branches and where all loops
have been fully unrolled.

This approach leads to highly optimised timestepper functions,
but, on the other hand, it can result in long compilation times
and high memory usage for large ODE systems. Thus, heyoka provides
also a *compact mode* option in which the code generation uses
more traditional idioms that greatly reduce compilation times
and memory usage. Compact mode results in a performance degradation
of :math:`\lesssim 2\times` with respect to the default code generation
mode, but it renders heyoka usable with ODE systems consisting
of thousands of terms.
