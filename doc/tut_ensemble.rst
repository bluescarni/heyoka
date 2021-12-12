Ensemble propagations
=====================

.. versionadded:: 0.17.0

Starting with version 0.17.0, heyoka offers support for
*ensemble propagations*. In ensemble mode, multiple distinct
instances of the same ODE system are integrated in parallel,
typically using different sets of initial conditions and/or
:ref:`runtime parameters <tut_param>`.
Monte Carlo simulations and parameter
searches are two typical examples of tasks in which ensemble mode
is particularly useful.

The ensemble mode API mirrors the time-limited propagation
functions available in the
:ref:`adaptive integrator class <tut_adaptive>`. Specifically,
three functions are available:

* ``ensemble_propagate_until()``, for ensemble propagations
  up to a specified epoch,
* ``ensemble_propagate_for()``, for ensemble propagations
  for a time interval,
* ``ensemble_propagate_grid()``, for ensemble propagations
  over a time grid.

In this tutorial, we will be focusing on the ``ensemble_propagate_until()``
function, but adapting the code to the other two functions
should be straightforward.

Note that, at this time, ensemble propagations are limited to using multiple threads
of execution to achieve parallelisation. In the future, additional parallelisation
modes (e.g., multiprocessing, distributed) may be added.

A simple example
----------------

As usual, for this illustrative tutorial we will be using the ODEs of the simple pendulum.
Thus, let us begin
with the definition of the symbolic variables and of an integrator object:

.. literalinclude:: ../tutorial/ensemble.cpp
    :language: c++
    :lines: 17-28

Note how, differently from the other tutorials, here we have set the initial conditions
to zeroes. This is because, in ensemble mode, we will never use directly the ``ta`` object to perform
a numerical integration. Rather, ``ta`` acts as a *template* from which other integrator
objects will be constructed, and thus its initial conditions are inconsequential.

The ``ensemble_propagate_until()`` function takes in input at least 4 arguments:

* the template integrator ``ta``,
* the final epoch ``t`` for the propagations (this argument would be a time interval ``delta_t``
  for ``ensemble_propagate_for()`` and a time grid for ``ensemble_propagate_grid()``),
* the number of iterations ``n_iter`` in the ensemble,
* a function object ``gen``, known as the *generator*.

The signature of the generator ``gen`` reads:

.. code-block:: c++

   taylor_adaptive<double> gen(taylor_adaptive<double> ta, std::size_t idx);

That is, the generator takes in input a *copy* of the template integrator ``ta``
and an iteration index ``idx`` in the ``[0, n_iter)`` range. ``gen`` is then expected
to modify the copy of ``ta`` (e.g., by setting its initial conditions to specific
values) and return it.

The ``ensemble_propagate_until()`` function iterates over
the ``[0, n_iter)`` range. At each iteration, the generator ``gen`` is invoked,
with the template integrator as first argument and the current iteration number 
as the second argument. The ``propagate_until()`` member
function is then called on the integrator returned by ``gen``, and the result of the propagation
is appended to a list of results which is finally returned by
``ensemble_propagate_until()`` once all the propagations have finished.

Let us see a concrete example of ``ensemble_propagate_until()`` in action. First, we
begin by creating 10 sets of different initial conditions to be used in the ensemble
propagations:

.. literalinclude:: ../tutorial/ensemble.cpp
    :language: c++
    :lines: 30-35

Next, we define a generator that will pick a set of initial conditions from ``ensemble_ics``,
depending on the iteration index:

.. literalinclude:: ../tutorial/ensemble.cpp
    :language: c++
    :lines: 37-43

We are now ready to invoke the ``ensemble_propagate_until()`` function:

.. literalinclude:: ../tutorial/ensemble.cpp
    :language: c++
    :lines: 45-46

Note that the ``ensemble_propagate_until()`` function can be invoked with additional keyword arguments (beside
the mandatory initial 4 arguments). Any additional argument will
be forwarded to each ``propagate_until()`` invocation.

The value returned by ``ensemble_propagate_until()`` is a vector of tuples constructed by concatenating the integrator
object used for each integration and the tuple returned by each ``propagate_until()`` invocation. This way, at the end
of an ensemble propagation it is possible to inspect both the state of each integrator object and the outcome of
each invocation of ``propagate_until()`` (including, e.g., the :ref:`continuous output <tut_c_output>`, if
requested).

For instance, we can print to screen the integrator used for the last iteration of the ensemble. This is the
first object of the ninth tuple in the return value:

.. literalinclude:: ../tutorial/ensemble.cpp
    :language: c++
    :lines: 48-50

.. code-block:: console

   Tolerance               : 2.2204460492503131e-16
   High accuracy           : false
   Compact mode            : false
   Taylor order            : 20
   Dimension               : 2
   Time                    : 20.000000000000000
   State                   : [0.12257736827306077, 0.24068377640981869]

The other values in the tuple are those returned by the ``propagate_until()`` invocation:

.. literalinclude:: ../tutorial/ensemble.cpp
    :language: c++
    :lines: 52-57

.. code-block:: console

   Integration outcome: taylor_outcome::time_limit
   Min/max timesteps  : 0.158147/0.167025
   N of timesteps     : 124
   Continuous output  : 0

Thread safety considerations
----------------------------

Because currently ensemble propagations always use thread-based parallelism,
it is important to ensure that all the objects involved in an ensemble propagation
provide a certain degree of thread safety. This concerns
especially the user-provided function objects such as the generator and
the callbacks (if present).

In an ensemble propagation, it is thus important to keep in mind that
the following actions may be performed
concurrently by separate threads of execution:

* invocation of the generator's call operator,
* copy construction of the callbacks (both the events' callbacks
  and the callbacks that can optionally be passed to the
  ``propagate_*()`` functions of the adaptive integrators),
* invocation of the call operators of the copies of the callbacks.

For instance, an event callback which performs write operations
on a global variable without using some form of synchronisation
will result in undefined behaviour when used in an ensemble propagation.
