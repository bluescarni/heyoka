.. _tut_param:

Runtime parameters
==================

.. versionadded:: 0.2.0

In all the examples we have seen so far, numerical constants
have always been hard-coded to fixed values when constructing
the expressions of the ODEs.

While this approach
leads to efficient code generation by embedding the values of the constants
directly in the source code and by opening up further
optimisation opportunities for the compiler, on the other hand it requires
the construction of a new integrator if the values of the numerical
constants change. In some applications (e.g., parametric studies),
this overhead can become a performance bottleneck, especially
for large ODE systems.

In order to avoid having to re-create a new integrator if the value of a constant
in an expression changes, heyoka's expression system provides a node type,
called :cpp:class:`~heyoka::param` (or *runtime parameter*), which
represents mathematical constants whose value is not known at the time of
construction of the expression.

In this section, we will illustrate
the usage of runtime parameters in heyoka via a pendulum
ODE system in which the values of the gravitational constant :math:`g`
and of the pendulum's length :math:`l` are left undetermined at the time
of the definition of the system:

.. math::

   \begin{cases}
   x^\prime = v \\
   v^\prime = -\frac{g}{l} \sin x
   \end{cases}.

Let's start with the construction of the integrator:

.. literalinclude:: ../tutorial/pendulum_param.cpp
   :language: c++
   :lines: 17-33

With respect to the previous examples, where :math:`g/l` had been
hard-coded to ``9.8``, now :math:`g/l` is represented as ``par[0] / par[1]``.
The syntax ``par[i]`` indicates a runtime parameter that is stored
at the index ``i`` in an array of parameter values. The array of
parameter values is optionally passed to the constructor as the :ref:`keyword argument <kwargs>`
``kw::pars`` (which, in this case, contains the values ``9.8`` for :math:`g` and ``1.``
for :math:`l`).
If an array of parameter values is not passed to the constructor,
heyoka will infer that the ODE system contains 2 parameters and will then initialise
the array of parameter values with zeroes.

If we try to print the integrator to screen, the output will confirm the
presence of runtime parameters:

.. code-block:: console

   Tolerance               : 2.2204460492503131e-16
   Taylor order            : 20
   Dimension               : 2
   Time                    : 0.0000000000000000
   State                   : [0.050000000000000003, 0.0000000000000000]
   Parameters              : [9.8000000000000007, 1.0000000000000000]

Note that the array of parameter values, like the state vector and the time coordinate,
is stored as a data member in the integrator object.

The period of a pendulum of length :math:`1\,\mathrm{m}` on Earth is :math:`\sim 2\,\mathrm{s}`
in the small-angle approximation,
as it can be confirmed via numerical integration:

.. literalinclude:: ../tutorial/pendulum_param.cpp
   :language: c++
   :lines: 38-40

.. code-block:: console

   Tolerance               : 2.2204460492503131e-16
   Taylor order            : 20
   Dimension               : 2
   Time                    : 2.0074035758801299
   State                   : [0.050000000000000003, 7.5784060331002885e-17]
   Parameters              : [9.8000000000000007, 1.0000000000000000]

As you can see, after 1 period the state of the system went back to the initial conditions.

We are now going to move to Mars, where the gravitational acceleration on the surface
is :math:`\sim 3.72\,\mathrm{m}/\mathrm{s}^2` (instead of Earth's
:math:`\sim 9.8\,\mathrm{m}/\mathrm{s}^2`). First we reset the time
coordinate:

.. literalinclude:: ../tutorial/pendulum_param.cpp
   :language: c++
   :lines: 42-43

Then we change the value of the gravitational constant :math:`g`, which,
as explained above, is stored at index 0 in the array of parameter values:

.. literalinclude:: ../tutorial/pendulum_param.cpp
   :language: c++
   :lines: 45-46

Note that, like the for the state data, the ``get_pars_data()``
function returns a naked pointer
that can be used to modify the parameter values. Another function
of the integrator object, ``get_pars()``,
returns a const reference to the ``std::vector``
holding the parameter values.

Because gravity is weaker on Mars, the period of a :math:`1\,\mathrm{m}` pendulum increases to
:math:`\sim 3.26\,\mathrm{s}`. We can confirm this via numerical integration:

.. literalinclude:: ../tutorial/pendulum_param.cpp
   :language: c++
   :lines: 48-50

.. code-block:: console

   Tolerance               : 2.2204460492503131e-16
   Taylor order            : 20
   Dimension               : 2
   Time                    : 3.2581889116828258
   State                   : [0.050000000000000003, 2.1864533707994132e-16]
   Parameters              : [3.7200000000000002, 1.0000000000000000]

Full code listing
-----------------

.. literalinclude:: ../tutorial/pendulum_param.cpp
   :language: c++
   :lines: 9-
