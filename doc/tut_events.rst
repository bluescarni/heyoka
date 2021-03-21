.. _tut_events:

Event detection
===============

.. versionadded:: 0.6.0

When integrating systems of ODEs, the need often arises to detect the occurrence of specific
conditions (or *events*) in the state of the system. Many real systems, for instance, are described by equations
that change discontinuously in response to particular conditions
(e.g., a spacecraft entering the cone of shadow of a planet,
or a thermostat switching on once the temperature reaches a certain level). In other situations,
detection of specific system states may suffice (e.g., in the computation of
`Poincaré sections <https://en.wikipedia.org/wiki/Poincar%C3%A9_map>`__).

An event in a system of ODEs can be defined by an *event equation* of the form 

.. math::

    g\left( t, \boldsymbol{x} \left( t \right) \right) = 0,

where, as usual, :math:`t` is the independent variable (time) and :math:`\boldsymbol{x} \left( t \right)` the state vector of the system.
As a concrete example, the collision between two spheres of radius 1 moving in a three-dimensional space can be described
by the event equation

.. math::

    \left( x_1 - x_0 \right)^2 + \left( y_1 - y_0 \right)^2 + \left( z_1 - z_0 \right)^2 - 4 = 0,

where :math:`\left( x_0, y_0, z_0 \right)` and :math:`\left( x_1, y_1, z_1 \right)` are the Cartesian coordinates
of the spheres' centres.

heyoka features a flexible and accurate event detection framework in which the :ref:`expression system <tut_expression_system>`
can be used to formulate arbitrary event equations. The event equations are then added to the ODE system and
integrated together with the other equations, so that, at every timestep, a Taylor series expansion of the event equations
in powers of time is available. Polynomial root finding techniques :cite:`collins1976polynomial` are then employed
on the Taylor series of the event equations to accurately locate the time of occurrence of an event within the timestep.

Like many other ODE integration libraries, heyoka makes a fundamental distinction between two types of events, *terminal* and *non-terminal*.
We will begin with non-terminal events, as they are conceptually simpler.

Non-terminal events
-------------------

Non-terminal events are events that do not modify the state of an ODE system. That is, the occurrence of a non-terminal event does not
change the system's dynamics and it does not alter the state vector of the system. A typical use of non-terminal events is to detect and log
when the system reaches a particular state of interest (e.g., flagging close encounters between celestial bodies, detecting when
a velocity or coordinate is zero, etc.).

As an initial example, we will turn to our good ole friend, the simple pendulum:

.. math::

    \begin{cases}
    x^\prime = v \\
    v^\prime = -9.8 \sin x
    \end{cases}.

Our goal will be to detect when the bob reaches the point of maximum amplitude, which corresponds to the angular velocity
:math:`v` going to zero. In other words, out (very simple) event equation is

.. math::

    v = 0.

We begin, as usual, with the definition of the symbolic variables:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 17-18

Next, we create a vector into which we will log the times at which :math:`v = 0`:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 20-22

We can now proceed to create a non-terminal event:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 24-38

Non-terminal events are represented in heyoka by the ``nt_event`` class. Like the :ref:`adaptive integrator <tut_adaptive>`
class, ``nt_event`` is parametrised over the floating-point type we want to use for  event detection
(in this case, ``double``). The first mandatory argument for the construction of a non-terminal event is the left-hand
side of the event equation, which in this case is simply :math:`v`.

The second mandatory construction argument is a callback function that will be invoked when the event is detected.
The callback function can be a lambda, a regular function, or a function object - the only requirement is that the
callback is a callable object with the following signature:

.. code-block:: c++

   void (taylor_adaptive<double> &, double);

The first function argument is a mutable reference to the integrator object, while the second argument is the absolute time
at which the event was detected.

Because non-terminal event detection is performed at the end of an integration step,
when the callback is invoked the state and time of the integrator object are those *at the end* of the integration
step in which the event was detected. Note that when integrating an ODE system with events, the ``taylor_adaptive``
class ensures that the Taylor coefficients are always kept up to date (as explained in the tutorial about
:ref:`dense output <tut_d_output>`), and thus in the callback function it is always possible to use the ``update_d_output()``
function to compute the dense output at any time within the last timestep that was taken.

.. warning::

    The ``taylor_adaptive`` object is passed as a non-const reference only so that it is possible to call
    non-const functions on it (such as ``update_d_output()``). Do not try to assign a new integrator object
    from within the callback, as that will result in undefined behaviour.

In this specific case, we perform two actions in the callback:

- first, we compute the dense output at the event trigger time and print
  the value of the ``x`` coordinate,
- second, we append to ``zero_vel_times`` the trigger time.

We are now ready to create our first event-detecting integrator:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 40-50

The list of non-terminal events is passed to the constructor of the
integrator via the ``kw::nt_events`` keyword argument. Note how we
set up the initial conditions so that the bob is at rest at an
angle of amplitude :math:`0.05`.

Let us now integrate for a few time units and observe the screen output:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 55-56

.. code-block:: console

   Value of x when v is zero: -0.05
   Value of x when v is zero: 0.05
   Value of x when v is zero: -0.05
   Value of x when v is zero: 0.05
   Value of x when v is zero: -0.05

As expected, when the velocity of the bob goes to zero
the absolute value the :math:`x` angle corresponds to the
initial amplitude of :math:`0.05`.

Let us now print the event times:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 58-61

.. code-block:: console

   Event detection time: 0
   Event detection time: 1.003701787940065
   Event detection time: 2.00740357588013
   Event detection time: 3.011105363820196
   Event detection time: 4.014807151760261

We can see how the the initial condition :math:`v_0 = 0` immediately
and correctly triggers an event at :math:`t = 0`. Physically, we know that the time
interval between the events must be half the period :math:`T` of the pendulum,
which can be computed exactly via elliptic functions. With the specific
initial conditions of this example, :math:`T = 2.0074035758801299\ldots`, and
we can see from the event times printed to screen
how the event detection system was accurate to machine precision.

Event direction
^^^^^^^^^^^^^^^

By default, heyoka will detect all zeroes of the event equations regardless
of the *direction* of the zero crossing (i.e., the value of the time derivative
of the event equation at the zero). However, it is sometimes useful to tigger the detection
of an event only if its direction is positive or negative. Event direction is represented
in heyoka by the ``event_direction`` enum, whose values can be

- ``event_direction::any`` (the default),
- ``event_direction::positive`` (derivative > 0),
- ``event_direction::negative`` (derivative < 0).

An event's direction can be specified upon construction:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 63-72

In this specific case, constraining the event direction to be positive is equivalent
to detect :math:`v = 0` only when the pendulum reaches the maximum amplitude on the left.
Let us take a look at the event times:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 74-80

.. code-block:: console

   Event detection time: 0
   Event detection time: 2.00740357588013
   Event detection time: 4.014807151760261

Indeed, the event now triggers only 3 times (instead of 5), and the times confirm
that the event is detected only when :math:`v` switches from negative to positive, i.e.,
at :math:`t=0`, :math:`t=T` and :math:`t=2T`.

Multiple events
^^^^^^^^^^^^^^^

When multiple events trigger within the same timestep, heyoka will process them
in chronological order (or reverse chronological order if integrating backwards in time).

Terminal events
---------------

Full code listing
-----------------

.. literalinclude:: ../tutorial/event_basic.cpp
   :language: c++
   :lines: 9-
