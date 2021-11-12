.. _tut_adaptive:

The adaptive integrator
=======================

The ``taylor_adaptive`` class provides an easy-to-use interface to heyoka's
main capabilities. Objects of this class can be constructed from a system
of ODEs and a set of initial conditions (plus a number of optional configuration parameters
with - hopefully - sensible defaults). Member functions are provided to
propagate in time the state of the system, either step-by-step or by specifying
time limits.

Let's see how we can use ``taylor_adaptive`` to integrate the ODE
system of the `simple pendulum <https://en.wikipedia.org/wiki/Pendulum_(mathematics)>`__,

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

   Tolerance               : 2.2204460492503131e-16
   Taylor order            : 20
   Dimension               : 2
   Time                    : 0.0000000000000000
   State                   : [0.050000000000000003, 0.025000000000000001]

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

   Outcome : taylor_outcome::success
   Timestep: 0.216053
   
   Tolerance               : 2.2204460492503131e-16
   Taylor order            : 20
   Dimension               : 2
   Time                    : 0.21605277478009474
   State                   : [0.043996448369926382, -0.078442455470687983]

It is also possible to perform a single timestep backward in time
via the ``step_backward()`` function:

.. literalinclude:: ../tutorial/adaptive_basic.cpp
   :language: c++
   :lines: 43-48

.. code-block:: console

   Outcome : taylor_outcome::success
   Timestep: -0.213123

The ``step()`` function can also be called with an argument representing
the maximum step size ``max_delta_t``: if the adaptive timestep
selected by heyoka is larger (in absolute value) than ``max_delta_t``,
then the timestep will be clamped to ``max_delta_t``:

.. literalinclude:: ../tutorial/adaptive_basic.cpp
   :language: c++
   :lines: 50-64

.. code-block:: console

   Outcome : taylor_outcome::time_limit
   Timestep: 0.01

   Outcome : taylor_outcome::time_limit
   Timestep: -0.02

Note that the integration outcome is now ``taylor_outcome::time_limit``, instead of ``taylor_outcome::success``.

Before moving on, we need to point out an important caveat when using the single
step functions:

.. warning::

   If the exact solution of the ODE system can be expressed as a polynomial function
   of time, the automatic timestep deduction algorithm may return a timestep of infinity.
   This is the case, for instance, when integrating the rectilinear motion of a free
   particle or the constant-gravity free-fall equation. In such cases, the step functions
   should be called with a finite ``max_delta_t`` argument, in order to clamp the timestep
   to a finite value.

   Note that the ``propagate_*()`` functions (described :ref:`below <tlimited_prop>`)
   are not affected by this issue.

Accessing state and time
------------------------

It is possible to read from and write to both the time variable and the state
vector. The ``get_time()``/``set_time()`` functions can be used to access
the time variable, while the ``get_state()`` and ``get_state_data()``
can be used to access the state data:

.. literalinclude:: ../tutorial/adaptive_basic.cpp
   :language: c++
   :lines: 66-75

Note that ``get_state()`` returns a const reference to the ``std::vector``
holding the integrator state, while ``get_state_data()`` returns a naked pointer
to the state data. Only ``get_state_data()`` can be used to mutate the state.

.. _tlimited_prop:

Time-limited propagation
------------------------

In addition to the step-by-step integration functions,
``taylor_adaptive`` also provides functions to propagate
the state of the system for a specified amount of time.
These functions are called ``propagate_for()`` and
``propagate_until()``: the former integrates
the system for a specified amount of time, the latter
propagates the state up to a specified epoch.

Let's see a couple of usage examples:

.. literalinclude:: ../tutorial/adaptive_basic.cpp
   :language: c++
   :lines: 77-93

.. code-block:: console

   Outcome      : taylor_outcome::time_limit
   Min. timestep: 0.202133
   Max. timestep: 0.218136
   Num. of steps: 24
   Current time : 5

   Outcome      : taylor_outcome::time_limit
   Min. timestep: 0.202122
   Max. timestep: 0.218139
   Num. of steps: 72
   Current time : 20

The time-limited propagation functions return
a tuple of 5 values, which represent, respectively:

* the outcome of the integration (which will always be
  ``taylor_outcome::time_limit``, unless error conditions arise),
* the minimum and maximum integration timesteps
  that were used in the propagation,
* the total number of steps that were taken,
* the :ref:`continuous output <tut_c_output>` function object,
  if requested (off by default).

The time-limited propagation functions can be used
to propagate both forward and backward in time:

.. literalinclude:: ../tutorial/adaptive_basic.cpp
   :language: c++
   :lines: 95-104

.. code-block:: console

   Outcome      : taylor_outcome::time_limit
   Min. timestep: 0.202078
   Max. timestep: 0.21819
   Num. of steps: 97
   Current time : 0

   Tolerance               : 2.2204460492503131e-16
   Taylor order            : 20
   Dimension               : 2
   Time                    : 0.0000000000000000
   State                   : [0.050000000000000044, 0.024999999999999991]

Note also that the time-limited propagation functions will stop
integrating if a non-finite value is detected in the state vector
at the end of the timestep. In such case, the outcome of the
integration will be ``taylor_outcome::err_nf_state``.

.. versionadded:: 0.7.0

The ``propagate_for()`` and ``propagate_until()`` functions
can be invoked with two additional optional keyword arguments:

- ``max_delta_t``: similarly to the ``step()`` function, this value
  represents the maximum timestep size in absolute value;
- ``callback``: this is a callable with signature

  .. code-block:: c++

     bool (taylor_adaptive<double> &);

  which will be invoked at the end of each timestep, with the integrator
  object as only argument. If the callback returns ``true`` then the integration
  will continue after the invocation of the callback, otherwise the integration
  will be interrupted.

Propagation over a time grid
----------------------------

.. versionadded:: 0.4.0

Another way of propagating the state of a system in a ``taylor_adaptive``
integrator is over a time grid. In this mode, the integrator
uses :ref:`dense output <tut_d_output>` to compute the state of the system
over a grid of time coordinates provided by the user. If the grid is denser
than the typical timestep size, this can be noticeably more efficient than
repeatedly calling ``propagate_until()`` on the grid points, because
propagating the system state via dense output is much faster than taking
a full integration step.

Let's see a simple usage example:

.. literalinclude:: ../tutorial/adaptive_basic.cpp
   :language: c++
   :lines: 106-113

``propagate_grid()`` takes in input a grid of time points represented as a ``std::vector``,
and returns a tuple of 5 values. The first 4 values are the same
as in the other ``propagate_*()`` functions:

* the outcome of the integration,
* the minimum and maximum integration timesteps
  that were used in the propagation,
* the total number of steps that were taken.

The fifth value returned by ``propagate_grid()`` is a ``std::vector`` containing
the state of the system at the time points in the grid. The state vectors are stored
contiguously in row-major order:

.. literalinclude:: ../tutorial/adaptive_basic.cpp
   :language: c++
   :lines: 115-117

.. code-block:: console

   x(0.4) = 0.0232578
   v(0.4) = -0.14078

There are no special requirements on the time values in the grid (apart from the
fact that they must be finite and ordered monotonically).

.. versionadded:: 0.7.0

The ``propagate_grid()`` function
can be invoked with two additional optional keyword arguments:

- ``max_delta_t``: similarly to the ``step()`` function, this value
  represents the maximum timestep size in absolute value;
- ``callback``: this is a callable with signature

  .. code-block:: c++

     bool (taylor_adaptive<double> &);

  which will be invoked at the end of each timestep, with the integrator
  object as only argument. If the callback returns ``true`` then the integration
  will continue after the invocation of the callback, otherwise the integration
  will be interrupted.

Full code listing
-----------------

.. literalinclude:: ../tutorial/adaptive_basic.cpp
   :language: c++
   :lines: 9-
