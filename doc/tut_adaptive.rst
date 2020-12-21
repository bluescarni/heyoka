.. _tut_adaptive:

The adaptive integrator class
=============================

The ``taylor_adaptive`` class provides an easy-to-use interface to heyoka's
main capabilities. Objects of this class can be constructed from a system
of ODEs and a set of initial conditions (plus a number of optional parameters
with - hopefully - sensible defaults). Member functions are provided to
propagate in time the state of the system, either step-by-step or by specifying
time limits.

Let's see how we can use ``taylor_adaptive`` to integrate the ODE
system of the simple pendulum,

.. math::

   \begin{cases}
   x^\prime = v \\
   v^\prime = -9.8 \sin x
   \end{cases}

with initial conditions

.. math::

   \begin{cases}
   x\left( 0 \right) = 0.05 \\
   v\left( 0 \right) = 0.025
   \end{cases}

Construction
------------

Here's the code:

.. literalinclude:: ../tutorial/adaptive_basic.cpp
   :language: c++
   :lines: 17-28

After creating the symbolic variables ``x`` and ``v``, we
construct an instance of ``taylor_adaptive<double>`` called ``ta``.
``taylor_adaptive`` is a class template parametrised over
the floating-point type which we want to use for the integration.
In this case, we use the ``double`` type, meaning that the integration
will be carried out in double precision.

As (mandatory) construction arguments, we pass in the system of
differential equations using the syntax ``prime(x) = ...``, and a set
of initial conditions for ``x`` and ``v`` respectively.

Let's try to print to screen the integrator object:

.. literalinclude:: ../tutorial/adaptive_basic.cpp
   :language: c++
   :lines: 31

This will produce the following output:

.. code-block:: console

   Taylor order: 20
   Dimension   : 2
   Time        : 0.0000000000000000
   State       : [0.050000000000000003, 0.025000000000000001]

By default, the error tolerance of an adaptive integrator is set to the
machine epsilon, which, for ``double``, is :math:`\sim 2.2\times10^{-16}`.
From this value, heyoka deduces an optimal Taylor order of 20, as indicated
by the screen output. ``taylor_adaptive`` manages its own copy of the state vector and the time variable.
Both the state vector and the time variable are updated automatically by the timestepping
functions. Note also how, by default, the time variable is initially set to zero.

Single timestep
---------------

Let's now try to perform a single integration timestep:

.. literalinclude:: ../tutorial/adaptive_basic.cpp
   :language: c++
   :lines: 33-41

First, we invoke the ``step()`` member function, which returns a pair of values.
The first value is a status flag indicating the outcome of the integration timestep,
while the second value is the step size that was selected by heyoka in order
to respect the desired error tolerance. We print both to screen, and we also
print again the ``ta`` object in order to inspect how state and time have changed.
The screen output will look something like this:

.. code-block:: console

   Status  : success
   Timestep: 0.216053
   
   Taylor order: 20
   Dimension   : 2
   Time        : 0.21605277478009474
   State       : [0.043996448369926382, -0.078442455470687983]

Accessing state and time
------------------------

It is possible to read from and write to both the time variable and the state
vector. The ``get_time()``/``set_time()`` functions can be used to access
the time variable, while the ``get_state()`` and ``get_state_data()``
can be used to access the state data.
