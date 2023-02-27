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
uses an error tolerance equal to the machine
epsilon of the floating-point type in use. E.g., when using the
``double`` floating-point type, the tolerance is set to
:math:`\sim 2.2\times 10^{-16}`.

The tolerance value is used by the ``taylor_adaptive``
class to control the error arising from truncating the (infinite)
Taylor series representing the solution of the ODE system.
In other words, ``taylor_adaptive`` strives to ensure that the
magnitude of the remainders of the Taylor series is
not greater than the tolerance,
either in an absolute or relative sense. Absolute error control mode
is activated when all elements of the state vector have a magnitude
less than 1, while relative error control mode is activated when at least one
element of the state vector has a magnitude greater than 1.

In order to specify a non-default tolerance, the keyword argument
``tol`` can be used when constructing an integrator object:

.. literalinclude:: ../tutorial/adaptive_opt.cpp
   :language: c++
   :lines: 19-36

.. code-block:: console

   Tolerance               : 1.0000000000000001e-09
   Taylor order            : 12
   Dimension               : 2
   Time                    : 0.0000000000000000
   State                   : [0.050000000000000003, 0.025000000000000001]

The optimal Taylor order for a tolerance of :math:`10^{-9}`
is now 12 (instead of 20 for a tolerance
of :math:`\sim 2.2\times 10^{-16}`).

Integrating the system back and forth shows how the accuracy of the
integration is reduced with respect to the default tolerance value:

.. literalinclude:: ../tutorial/adaptive_opt.cpp
   :language: c++
   :lines: 38-42

.. code-block:: console

   Tolerance               : 1.0000000000000001e-09
   Taylor order            : 12
   Dimension               : 2
   Time                    : 0.0000000000000000
   State                   : [0.050000000001312848, 0.024999999997558649]

.. _tut_compact_mode:

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
also a *compact mode* option in which code generation employs
more traditional programming idioms that greatly reduce compilation time
and memory usage. Compact mode results in a performance degradation
of :math:`\lesssim 2\times` with respect to the default code generation
mode, but it renders heyoka usable with ODE systems consisting
of thousands of terms.

Let's try to quantify the performance difference in a concrete case.
In this example, we first construct the ODE system corresponding
to an N-body problem with 6 particles via the ``model::nbody()``
utility function:

.. literalinclude:: ../tutorial/adaptive_opt.cpp
   :language: c++
   :lines: 46-47

Next, we create an initial state vector for our system.
The contents of the vector do not matter at this stage:

.. literalinclude:: ../tutorial/adaptive_opt.cpp
   :language: c++
   :lines: 49-50

Next, we time the creation of an integrator object in default
code generation mode:

.. literalinclude:: ../tutorial/adaptive_opt.cpp
   :language: c++
   :lines: 52-59

.. code-block:: console

   Default mode timing: 3807ms

Finally, we time the creation of the same integrator object
in compact mode (which can be activated via the ``compact_mode``
keyword argument):

.. literalinclude:: ../tutorial/adaptive_opt.cpp
   :language: c++
   :lines: 61-68

.. code-block:: console

   Compact mode timing: 269ms

That is, in this specific example compact mode is more than 10 times
faster than the default
code generation mode when it comes to the construction of the integrator
object. For larger ODE systems, the gap will be even wider.

High-accuracy mode
------------------

For long-term integrations at very low error tolerances, heyoka offers
an opt-in *high-accuracy* mode. In high-accuracy mode, heyoka
employs techniques that minimise the numerical errors arising from
the use of finite-precision floating-point numbers, at the cost
of a slight runtime performance degradation.

Currently, high-accuracy mode changes the way heyoka evaluates
the Taylor polynomials used to update the state of the system
at the end of an integration timestep. Specifically, polynomial evaluation
via Horner's rule is replaced by
`compensated summation <https://en.wikipedia.org/wiki/Kahan_summation_algorithm>`__,
which prevents catastrophic cancellation issues and ultimately helps maintaining
machine precision over very long integrations.

High-accuracy mode can be enabled via the ``high_accuracy`` keyword
argument.
