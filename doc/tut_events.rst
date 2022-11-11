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

   void (taylor_adaptive<double> &, double, int);

The first argument is a mutable reference to the integrator object, the second argument is the absolute time
at which the event was detected (i.e., the trigger time), and the last argument is the sign of the derivative
of the event equation at the trigger time (-1 for negative derivative, 1 for positive derivative and 0 for
zero derivative).

.. warning::

    The ``taylor_adaptive`` object is passed as a non-const reference only so that it is possible to call
    non-const functions on it (such as ``update_d_output()``). Do not try to assign a new integrator object
    from within the callback, as that will result in undefined behaviour.

Because non-terminal event detection is performed at the end of an integration step,
when the callback is invoked the state and time of the integrator object are those **at the end** of the integration
step in which the event was detected. Note that when integrating an ODE system with events, the ``taylor_adaptive``
class ensures that the Taylor coefficients are always kept up to date (as explained in the tutorial about
:ref:`dense output <tut_d_output>`), and thus in the callback function it is always possible to use the ``update_d_output()``
function to compute the dense output at any time within the last timestep that was taken.

In this example, we perform two actions in the callback:

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
of the event equation at the zero). However, it is sometimes useful to trigger the detection
of an event only if its direction is positive or negative. Event direction is represented
in heyoka by the ``event_direction`` enum, whose values can be

- ``event_direction::any`` (the default),
- ``event_direction::positive`` (derivative > 0),
- ``event_direction::negative`` (derivative < 0).

Event direction can be specified upon construction via the ``kw::direction`` keyword:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 63-72

In this specific case, constraining the event direction to be positive is equivalent
to detecting :math:`v = 0` only when the pendulum reaches the maximum amplitude on the left.
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

When multiple events trigger within the same timestep (or if the same event triggers
multiple times), heyoka will process the events in chronological order
(or reverse chronological order when integrating backwards in time).

Let us demonstrate this with another example with the simple pendulum.
We will now aim to detect two events defined by the equations:

.. math::

    \begin{cases}
    v = 0 \\
    v^2 - 10^{-12} = 0
    \end{cases}.

In other words, we are looking to determine the time of maximum amplitude (:math:`v = 0`) and
the time at which the absolute value of the angular velocity is small but not zero. Because
of the closeness of these events, we can expect both events to be detected during the same timestep,
with the second event triggering twice.

Let's begin by defining the two events:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 82-88

This time the events' callbacks just print the event time to screen, without
modifying the ``zero_vel_times`` list.

We can then reset the integrator, propagate for a few time units and check the screen output:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 90-94

.. code-block:: console

    Event 0 triggering at t=0
    Event 1 triggering at t=2.041666914753826e-06
    Event 1 triggering at t=1.003699746272244
    Event 0 triggering at t=1.003701787940065
    Event 1 triggering at t=1.003703829606799
    Event 1 triggering at t=2.007401534213656
    Event 0 triggering at t=2.00740357588013
    Event 1 triggering at t=2.00740561754654
    Event 1 triggering at t=3.011103322152711
    Event 0 triggering at t=3.011105363820196
    Event 1 triggering at t=3.011107405487484
    Event 1 triggering at t=4.014805110093445
    Event 0 triggering at t=4.014807151760261
    Event 1 triggering at t=4.014809193427102

Note how the events are indeed processed in chronological order, and how the event detection system is able to
successfully recognize the second event triggering twice in close succession.

Terminal events
---------------

The fundamental characteristic of terminal events is that, in contrast to non-terminal events,
they alter the dynamics and/or the state of the system. A typical example of a terminal event is the
`elastic collision <https://en.wikipedia.org/wiki/Elastic_collision>`__ of
two rigid bodies, which instantaneously and discontinuously changes the bodies' velocity vectors.
Another example is the switching on of a spacecraft engine, which alters the differential
equations governing the dynamics of the spacecraft.

Terminal events are represented in heyoka by the ``t_event`` class. Similarly to
the ``nt_event`` class, the construction of a ``t_event`` requires
at the very least the expression corresponding to the left-hand side of the event equation.
A number of additional optional keyword arguments can be passed to customise the behaviour
of a terminal event:

- ``kw::callback``: a callback function that will be called when the event triggers. Note that,
  for terminal events, the presence of a callback is optional (whereas it is mandatory for
  non-terminal events);
- ``kw::cooldown``: a floating-point value representing the cooldown time for the terminal event
  (see :ref:`below <tut_t_event_cooldown>` for an explanation);
- ``kw::direction``: a value of the ``event_direction`` enum which, like for non-terminal
  events, can be used to specify that the event should be detected only for a specific direction
  of the zero crossing.

It is important to understand how heyoka reacts to terminal events. At every integration timestep, heyoka
performs event detection for both terminal and non-terminal events. If one or more terminal events
are detected, heyoka will sort the detected terminal events by time and will select the first
terminal event triggering in chronological order (or reverse chronological order when integrating
backwards in time). All the other terminal events and all the non-terminal events triggering *after*
the first terminal event are discarded. heyoka then propagates the state of the system up to the
trigger time of the first terminal event, executes the callbacks of the surviving non-terminal events
in chronological order and finally executes the callback of the first terminal event (if provided).

In order to illustrate the use of terminal events, we will consider a damped pendulum with a small twist:
the friction coefficient :math:`\alpha` switches discontinuously between 1 and 0 every time the angular
velocity :math:`v` is zero. The ODE system reads:

.. math::

    \begin{cases}
    x^\prime = v \\
    v^\prime = - 9.8\sin x - \alpha v
    \end{cases},

and the terminal event equation is, again, simply :math:`v = 0`.

Let us begin with the definition of the terminal event:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 96-114

Like in the case of non-terminal events, we specified as first construction argument
the event equation. As second argument we passed a callback function that will be invoked
when the event triggers.

As you can see from the code snippet, the callback signature for terminal events
differs from the signature non-terminal callbacks. Specifically:

- the event trigger time is not passed to the callback. This is not necessary
  because, when a terminal event triggers, the state of the integrator is propagated
  up to the event, and thus the trigger time is the current integrator time
  (which can be fetched via ``ta.get_time()``);
- there is an additional boolean function argument, here called ``mr``. We will be ignoring
  this extra argument for the moment, its meaning will be clarified in the
  :ref:`cooldown section <tut_t_event_cooldown>`;
- whereas non-terminal event callbacks do not return anything, terminal event callbacks
  are required to return ``true`` or ``false``. If the callback returns ``false`` the integration
  will always be stopped after the execution of the callback. Otherwise, when using the
  ``propagate_*()`` family of functions, the integration will resume after the execution
  of the callback.

Note that, for the purpose of stopping the integration, an event *without* a callback is considered
equivalent to an event whose callback returns ``false``.
We thus refer to terminal events without a callback or whose callback returns ``false``
as *stopping* terminal events, because their occurrence will prevent the integrator from continuing
without user intervention.

Like for non-terminal events, the last callback argument is the sign of the time derivative
of the event equation at the event trigger time.

In this example, within the callback code we alter the value of the drag coefficient :math:`\alpha`
(which is stored within the :ref:`runtime parameters <tut_param>` of the integrator): if :math:`\alpha`
is currently 0, we set it to 1, otherwise we set it to 0.

Let us proceed to the construction of the integrator:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 116-124

Similarly to the non-terminal events case, the list of terminal events
is specified when constructing an integrator via the ``kw::t_events`` keyword argument.

If a terminal event triggers within the single-step functions (``step()`` and ``step_backward()``),
the outcome of the integration will contain the index of the event that triggered. Let us see a simple example:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 126-134

.. code-block:: console

   Integration outcome: taylor_outcome::terminal_event_0 (continuing)
   Event index        : 0

The screen output confirms that the first (and only) event triggered. For stopping terminal events,
the numerical value of the outcome is the opposite of the event index minus one.

Because here we used the single step
function, even if the event's callback returned ``true`` the integration was stopped in correspondence of the
event. Let us now use the ``propagate_grid()`` function instead, so that the integration resumes after the
execution of the callback:

.. literalinclude:: ../tutorial/event_basic.cpp
    :language: c++
    :lines: 136-145

.. code-block:: console

   [-0.02976504606251412, -0.02063006479837935]
   [0.02970666582653454, 0.02099345736431702]
   [-0.01761378049610636, -0.01622382722426959]
   [0.01757771112979705, 0.01613903817360225]
   [-0.01037481471383597, -0.01205316233867281]
   [0.01035648925410416, 0.01177669636844242]
   [-0.006080605964468329, -0.008627473720971276]
   [0.006074559637531474, 0.008299135527482404]
   [-0.003544733998720797, -0.006013682818278612]
   [0.003546198899884463, 0.005703010459398463]

   Final time: 10

The screen output confirms that indeed the integration continued up to the final time :math:`t = 10`. The values
of the angle and angular velocity throughout the integration show how the drag coefficient (which was on roughly
for half of the total integration time) progressively slowed the bob down.

.. _tut_t_event_cooldown:

Cooldown
^^^^^^^^

One notable complication when restarting an integration that was stopped in correspondence of a terminal event
is the risk of immediately re-triggering the same event, which would lead to an endless loop without any progress
being made in the integration. This phenomenon is sometimes called *discontinuity sticking* in the literature.

In order to avoid this issue, whenever a terminal event occurs the event enters
a *cooldown* period. Within the cooldown period, occurrences of the same event are ignored by the event detection
system.

The length of the cooldown period is, by default, automatically deduced by heyoka, following a heuristic
that takes into account:

- the error tolerance of the integrator,
- the derivative of the event equation at the trigger time.

The heuristic works best under the assumption that the event function :math:`g\left( t, \boldsymbol{x} \left( t \right) \right)`
does not change (much) after the execution of the event's callback. If, for any reason, the automatic deduction heuristic is
to be avoided, it is possible to set a custom value for the cooldown.
A custom cooldown period can be selected when constructing
a terminal event via the ``kw::cooldown`` keyword argument.

When a terminal event triggers and enters the cooldown period, the event detection system will also try to detect
the occurrence of multiple roots of the event equation within the cooldown period. If such multiple roots are detected,
then the ``mr`` boolean parameter in the terminal event callback will be set to ``true``, so that the user
has the possibility to handle such occurrence. Note that an ``mr`` value of ``false`` in the callback does not imply
that multiple roots do not exist, just that they were not detected.

Note that manually modifying the integrator's time or state does **not** automatically reset the cooldown values
for terminal events. This could in principle lead to missing terminal events when the integration restarts.
For this reason, a member function called ``reset_cooldowns()`` is available to clear the cooldown timers of
all terminal events.

Limitations and caveats
-----------------------

Badly-conditioned event equations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because heyoka's event detection system is based on polynomial root finding techniques, it will experience
issues when the Taylor series of the event equations have roots of multiplicity greater than 1. This is usually
not a problem in practice, unless the event equations are written in such a way to always generate polynomials
with multiple roots.

For instance, an event equation such as

.. math::

    \left[ g\left( t, \boldsymbol{x} \left( t \right) \right) \right]^2 = 0

will be troublesome, because both the event equation *and* its time derivative will be zero
when the event triggers. This will translate to a Taylor series with a double root in correspondence
of the event trigger time, which will lead to a breakdown of the root finding algorithm.
This, at best, will result in reduced performance and, at worst, in missing events altogether.
Additionally, in case of terminal events the automatically-deduced cooldown value in correspondence of
a double root will tend to infinity.

As a general rule, users should then avoid defining event equations in which the event trigger times
are stationary points.

Note that missed events due to badly-conditioned polynomials will likely be flagged by heyoka's logging system.

Event equations and timestepping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As explained earlier, the differential equations of the events are added to the ODE system and
integrated together with the original equations. Because of this, event equations influence the
selection of the adaptive timestep, even if no event is ever detected throughout the integration.

For instance, the absolute value of the event equation at the beginning of the timestep is taken
into account for the determination of the timestep size in relative error control mode. Thus, if
the typical magnitude of the event equation throughout the integration is much larger than the typical
magnitude of the state variables, the integration error for the state variables will increase with respect
to an integration without event detection.

As another example, an event equation which requires small timesteps for accurate numerical propagation
will inevitably slow down also the propagation of the ODEs.


Full code listing
-----------------

.. literalinclude:: ../tutorial/event_basic.cpp
   :language: c++
   :lines: 9-
