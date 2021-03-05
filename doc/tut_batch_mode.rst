.. _tut_batch_mode:

Batch mode
==========

heyoka's API supports a mode of operation called *batch mode*.
In batch mode, all the scalar quantities appearing in a system of ODEs
(i.e., state variables, time coordinate, parameters, etc.)
are formally replaced by small vectors of fixed size :math:`n`, so that,
effectively, multiple ODE systems sharing the same mathematical formulation
are being integrated simultaneously using different sets of numerical values.

Because modern CPUs support `SIMD instructions <https://en.wikipedia.org/wiki/SIMD>`__,
the runtime cost of operating on a vector of :math:`n` scalar values is roughly
equivalent to the cost of operating on a single scalar value, and thus the use of
batch mode can lead to an increase in floating-point throughput up to a factor of :math:`n`.

It is important to emphasise that batch mode does not reduce
the CPU time required to integrate a system of ODEs. Rather, as a fine-grained
form of data parallelism, batch mode allows to integrate multiple ODE systems in parallel
at no additional cost, and it is thus most useful when the need arise
to integrate the same ODE system with different initial conditions and parameters.

Although batch mode can in principle be used with all floating-point types supported
by heyoka, in practice at this time no CPU provides SIMD instructions for extended-precision
datatypes. Thus, here we will consider the application of batch mode only to
standard ``double`` precision computations.

The value of the batch size :math:`n` can be freely chosen by the user. In order
to achieve optimal performance, however, :math:`n` should match the SIMD width of the
processor in use. Because at this time the most widespread SIMD instruction set is
`AVX <https://en.wikipedia.org/wiki/Advanced_Vector_Extensions>`__ (available on
most x86 processors sold since 2011), in this tutorial we will be using a
batch size :math:`n=4`.

The adaptive batch integrator
-----------------------------

The ``taylor_adaptive_batch`` class is the batch mode counterpart of the adaptive
(scalar) integrator :ref:`described earlier <tut_adaptive>`. Although at a high-level
the API of ``taylor_adaptive_batch`` is quite similar to the API of
``taylor_adaptive``, there are also some important differences that need to be
pointed out.

In order to present a comprehensive example, we will consider again the integration
of the :ref:`forced damped pendulum <tut_nonauto>`, with a small modification:

.. math::

   \begin{cases}
   x^\prime = v \\
   v^\prime = \cos t - \alpha v - \sin(x)
   \end{cases}.

Here :math:`\alpha` is an air friction coefficient whose value is left undefined
(i.e., :math:`\alpha` is a :ref:`runtime parameter <tut_param>`).

Let us take a look at the first few lines of the C++ code:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 26-30

As usual, we begin by creating the symbolic state variables ``x`` and ``v``.
We also store in a constant the value of the batch size :math:`n = 4` for later use.

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 32-34

We then create two memory buffers to hold, respectively, the state of the
system and the values of the runtime parameter :math:`\alpha`. Because we are operating
in batch mode, we need to store :math:`2\times n` values for the state variables
and :math:`1\times n` values for the (only) runtime parameter.

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 36-39

Next, we create two `xtensor <https://xtensor.readthedocs.io/en/latest/>`__ adaptors
on the memory buffers that we just created. These adaptors will allow us
to index into the state and parameters vectors as if they were 2D arrays. Note that these
adaptors do not perform any copy of the original data, rather they just provide
an alternative view of the underlying memory buffers. Because heyoka requires elements
of a batch to be stored contiguously, the adaptors are set up with, respectively, 2 and 1
rows, and :math:`n` columns.

The next step is the setup of the initial conditions and of the values of the parameter :math:`\alpha`:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 41-48

Here we are using xtensor's indexing capabilities to assign the batch of initial values
:math:`x_0=\left( 0.01, 0.02, 0.03, 0.04 \right)` and
:math:`v_0=\left( 1.85, 1.86, 1.87, 1.88 \right)`, and the values for the parameter
:math:`\alpha = \left( 0.10, 0.11, 0.12, 0.13 \right)`. Note the the syntax
``xt::view(s_arr, 0, xt::all())`` would be equivalent, in NumPy, to ``s_arr[0, :]``.
We can verify that the values were correctly assigned by printing to screen the contents
of the arrays:

.. code-block:: console

   State array:
   {{ 0.01,  0.02,  0.03,  0.04},
    { 1.85,  1.86,  1.87,  1.88}}

   Parameters array:
   {{ 0.1 ,  0.11,  0.12,  0.13}}

We are now ready to create the integrator object:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 50-62

The constructor of ``taylor_adaptive_batch`` is very similar to the constructor of
``taylor_adaptive``: it has the same mandatory and optional arguments, plus an extra
mandatory argument representing the batch size. Note that it is important that
``state`` and ``pars`` are moved into the constructor (rather than merely copied),
so that the adaptors ``s_arr`` and ``p_arr`` will now refer to the state
and parameters vectors stored inside the integrator object.

Because we didn't provide values for the initial times, the time coordinates are
all initialised to zero, as we can verify by printing to screen the time array:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 64-67

.. code-block:: console

   Time array: { 0.,  0.,  0.,  0.}

We are now ready to start integrating. Like ``taylor_adaptive``, ``taylor_adaptive_batch``
provides ``step()`` functions for integrating forward or backward in time step-by-step.
One important difference is that, in order to avoid costly memory allocations,
the ``step()`` functions of the batch integrator do not return anything. Rather, the
batch integrator maintains an internal vector of outcomes which is updated at the
end of each timestep. Let's take a look:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 69-78

.. code-block:: console

   Batch index 0: (taylor_outcome::success, 0.205801)
   Batch index 1: (taylor_outcome::success, 0.20587)
   Batch index 2: (taylor_outcome::success, 0.204791)
   Batch index 3: (taylor_outcome::success, 0.203963)

We can see how the integration timestep was successful for all elements of the batch,
and how slightly different timesteps were chosen for each element of the batch.

Let's also print to screen the updated state and time arrays:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 80-81

.. code-block:: console

   State array:
   {{ 0.404885,  0.416439,  0.425714,  0.435479},
    { 1.973176,  1.976935,  1.980292,  1.983766}}

   Time array:
   { 0.205801,  0.20587 ,  0.204791,  0.203963}

Note that because the initial conditions were all set to similar values,
the state of the system after a single timestep also does not change much
across the batch elements.

Like for ``taylor_adaptive``, the ``step()`` function can be invoked with
a vector of time limits: if the adaptive timesteps
selected by heyoka are larger (in absolute value) than the specified limits,
then the timesteps will be clamped.

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 83-95

.. code-block:: console

   Batch index 0: (taylor_outcome::time_limit, 0.01)
   Batch index 1: (taylor_outcome::time_limit, 0.011)
   Batch index 2: (taylor_outcome::time_limit, 0.012)
   Batch index 3: (taylor_outcome::time_limit, 0.013)

   State array:
   {{ 0.424636,  0.438206,  0.449501,  0.461293},
    { 1.97695 ,  1.980738,  1.984087,  1.987488}}

   Time array:
   { 0.215801,  0.21687 ,  0.216791,  0.216963}

The ``propagate_for()`` and ``propagate_until()`` functions are also available for the
batch integrator. Similarly to the ``step()`` functions, the outcomes of the ``propagate_*()``
functions are stored in internal vectors of tuples, with the tuple elements representing:

* the outcome of the integration,
* the minimum and maximum integration timesteps
  that were used in the propagation,
* the total number of steps that were taken.

Let's see a couple of examples:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 97-115

.. code-block:: console

   Batch index 0: (taylor_outcome::time_limit, 0.197348, 0.428668, 34)
   Batch index 1: (taylor_outcome::time_limit, 0.191913, 0.429224, 38)
   Batch index 2: (taylor_outcome::time_limit, 0.188229, 0.433903, 41)
   Batch index 3: (taylor_outcome::time_limit, 0.184475, 0.464741, 44)

   State array:
   {{ 4.612543,  2.727621,  1.123953,  0.173771},
    {-2.246896, -1.917584, -1.783502, -1.11716 }}

   Time array:
   { 10.215801,  11.21687 ,  12.216791,  13.216963}

   Batch index 0: (taylor_outcome::time_limit, 0.204735, 0.307217, 40)
   Batch index 1: (taylor_outcome::time_limit, 0.211805, 0.317214, 38)
   Batch index 2: (taylor_outcome::time_limit, 0.224914, 0.410416, 35)
   Batch index 3: (taylor_outcome::time_limit, 0.213014, 0.371655, 34)

   State array:
   {{ 1.801537,  2.833631,  3.399033,  6.072237},
    { 1.36256 ,  0.503107, -0.06062 ,  0.81854 }}

   Time array:
   { 20.,  21.,  22.,  23.}

Full code listing
-----------------

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 9-
