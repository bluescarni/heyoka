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
and parameters vectors stored inside the integrator object. These vectors can
also be accessed directly using member functions such as ``get_state()``,
``get_state_data()``, ``get_pars()`` and ``get_pars_data()``, as showed
in previous tutorials.

Because we didn't provide values for the initial times, the time coordinates are
all initialised to zero, as we can verify by printing to screen the time array:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 64-67

.. code-block:: console

   Time array: { 0.,  0.,  0.,  0.}

Note that, contrary to the scalar integrator, in the batch integrator it is not
possible to write directly into the array of time coordinates. The function ``set_time()``,
accepting a ``std::vector`` of time coordinates in input, must be used instead.

Step-by-step integration
^^^^^^^^^^^^^^^^^^^^^^^^

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

Time-limited propagation
^^^^^^^^^^^^^^^^^^^^^^^^

The ``propagate_*()`` functions are also available for the
batch integrator. Similarly to the ``step()`` functions, the outcomes of the ``propagate_*()``
functions are stored in internal vectors of tuples, with the tuple elements representing:

* the outcome of the integration,
* the minimum and maximum integration timesteps
  that were used in the propagation,
* the total number of steps that were taken.

The ``propagate_for/until()`` functions in batch mode return the
:ref:`continuous output <tut_c_output_batch>` function object
(if requested). The ``propagate_grid()`` function returns
the result of the integration over a grid of time batches.

Let us see a couple of examples:

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

An important difference with respect to scalar mode is that the vector returned
by ``propagate_grid()`` in batch mode always contains values for *all* grid points,
event if the integration is terminated early (e.g., due to errors or to terminal
events triggering). In scalar mode, by contrast, in case of early termination
``propagate_grid()`` returns only the output values for the grid points that
could be computed before the early exit. In batch mode, the values for the grid
points that could not be reached due to early exit are filled with NaNs.

.. _tut_c_output_batch:

Dense & continuous output
^^^^^^^^^^^^^^^^^^^^^^^^^^

The batch integrator also supports :ref:`dense output <tut_d_output>`. Like for ``taylor_adaptive``,
enabling dense output is a two-step process. First we invoke one of the ``step()`` functions
with the optional flag set to ``True``. This will write the Taylor coefficients that were
used to propagate the last timestep into an internal vector. Let's also create an adaptor
over this vector for ease of indexing:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 117-131

.. code-block:: console

   Array of Taylor coefficients:
   {{{ 1.801537e+00,  2.833631e+00,  3.399033e+00,  6.072237e+00},
     { 1.362560e+00,  5.031073e-01, -6.062030e-02,  8.185404e-01},
     {-3.508356e-01, -4.530940e-01, -3.690403e-01, -2.149280e-01},
     {-8.852698e-02, -4.292286e-02,  6.466405e-03,  1.695101e-02},
     { 5.383037e-02, -8.782188e-03,  1.169218e-02,  3.331864e-02},
     {-2.257045e-02,  6.540328e-04, -3.247063e-04, -2.436106e-03},
     {-2.741537e-03,  1.561782e-03, -1.558171e-03, -4.247622e-03},
     { 5.787575e-03, -1.040219e-03,  1.346095e-04,  5.888381e-04},
     {-1.631527e-03,  1.131481e-04,  1.576557e-04,  3.288048e-04},
     {-5.334194e-04,  6.543342e-05, -9.001253e-06, -1.209411e-04},
     { 5.504048e-04, -8.585946e-06, -6.654560e-06, -8.597949e-06},
     {-1.163014e-04,  9.694202e-06,  3.026559e-07,  1.801841e-05},
     {-7.225493e-05, -2.188738e-06,  4.551510e-07, -3.366484e-06},
     { 5.707623e-05,  2.686672e-08, -4.619774e-08, -1.767741e-06},
     {-7.955589e-06,  2.877763e-07, -3.207529e-08,  8.982967e-07},
     {-9.701333e-06, -1.598994e-07,  1.713127e-10,  4.169540e-08},
     { 6.053724e-06,  2.748795e-08, -9.273869e-10, -1.395324e-07},
     {-3.664674e-07, -3.468361e-09,  5.812755e-10,  2.711640e-08},
     {-1.269841e-06, -3.437010e-09,  2.916397e-10,  1.403129e-08},
     { 6.459461e-07,  1.941421e-09, -6.667302e-11, -7.361741e-09},
     { 1.902250e-08, -7.307133e-10, -3.357923e-11, -3.211871e-10}},
    {{ 1.362560e+00,  5.031073e-01, -6.062030e-02,  8.185404e-01},
     {-7.016711e-01, -9.061880e-01, -7.380806e-01, -4.298560e-01},
     {-2.655809e-01, -1.287686e-01,  1.939922e-02,  5.085302e-02},
     { 2.153215e-01, -3.512875e-02,  4.676871e-02,  1.332746e-01},
     {-1.128522e-01,  3.270164e-03, -1.623531e-03, -1.218053e-02},
     {-1.644922e-02,  9.370692e-03, -9.349029e-03, -2.548573e-02},
     { 4.051302e-02, -7.281531e-03,  9.422663e-04,  4.121867e-03},
     {-1.305222e-02,  9.051852e-04,  1.261245e-03,  2.630438e-03},
     {-4.800774e-03,  5.889008e-04, -8.101127e-05, -1.088470e-03},
     { 5.504048e-03, -8.585946e-05, -6.654560e-05, -8.597949e-05},
     {-1.279316e-03,  1.066362e-04,  3.329215e-06,  1.982025e-04},
     {-8.670591e-04, -2.626486e-05,  5.461812e-06, -4.039781e-05},
     { 7.419910e-04,  3.492673e-07, -6.005706e-07, -2.298064e-05},
     {-1.113782e-04,  4.028868e-06, -4.490541e-07,  1.257615e-05},
     {-1.455200e-04, -2.398491e-06,  2.569691e-09,  6.254310e-07},
     { 9.685958e-05,  4.398072e-07, -1.483819e-08, -2.232518e-06},
     {-6.229946e-06, -5.896214e-08,  9.881684e-09,  4.609788e-07},
     {-2.285714e-05, -6.186618e-08,  5.249515e-09,  2.525632e-07},
     { 1.227298e-05,  3.688699e-08, -1.266787e-09, -1.398731e-07},
     { 3.804500e-07, -1.461427e-08, -6.715846e-10, -6.423741e-09},
     {-3.444138e-06,  2.634318e-09,  1.671414e-10,  2.478574e-08}}}

Quite a mouthful! Let's print to screen the order-0 Taylor coefficients
of ``x`` and ``v`` for all batch elements:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 133-134

.. code-block:: console

   Order-0 x: { 1.801537,  2.833631,  3.399033,  6.072237}
   Order-0 v: { 1.36256 ,  0.503107, -0.06062 ,  0.81854 }

Indeed, as expected the order-0 Taylor coefficients correspond to the initial
conditions at the beginning of the previous timestep (see earlier screen output).

After computing the Taylor coefficients, we can ask for the dense output
at different time coordinates:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 136-141

.. code-block:: console

   Dense output:
   {{ 1.934202,  2.879367,  3.389288,  6.151962},
    { 1.289941,  0.411166, -0.134188,  0.776195}}

.. versionadded:: 0.16.0

Support for :ref:`continuous output <tut_c_output>` is also available in batch mode.
Like in scalar mode, continuous output is requested via the ``c_output`` boolean keyword flag,
which can be passed to the ``propagate_for/until()`` functions. The usage of the
continuous output object returned by ``propagate_for/until()`` is analogous to the scalar case,
the only difference being that the call operator expects a batch of time coordinates
(represented as a ``std::vector``)
rather than a single scalar time corrdinate.


Event detection
^^^^^^^^^^^^^^^

.. versionadded:: 0.16.0

:ref:`Event detection <tut_events>` in also available in batch mode. The API is similar
to the scalar mode API, with the following differences:

* the event classes in batch mode are called ``nt_event_batch`` and ``t_event_batch``
  (for non-terminal and terminal events respectively), rather than ``nt_event`` and ``t_event``;
* with respect to scalar mode, the callback signatures in batch mode feature an extra trailing argument
  of type ``std::uint32_t`` that indicates in which element of the batch the event was detected.

Let us see a concrete example: we will be integrating in batch mode the simple pendulum with slightly
different sets of initial conditions for each batch element, and we want to detect via
a non-terminal event when the bob's velocity is zero.
When the event triggers, we will be printing to screen the time and the value of the angle
coordinate for the batch element in which the event triggered.

We begin with the creation of a non-terminal event in batch mode:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 143-156

Next, we create the integrator object:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 158-169

We can now propagate for a few time units:

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 171-173

.. code-block:: console

   Zero velocity time and angle for batch element 0: 0.501973, 0.0798808
   Zero velocity time and angle for batch element 1: 0.463715, 0.0836782
   Zero velocity time and angle for batch element 2: 0.429231, 0.0885657
   Zero velocity time and angle for batch element 3: 0.398675, 0.0943745
   Zero velocity time and angle for batch element 1: 1.4677, -0.0836782
   Zero velocity time and angle for batch element 2: 1.43327, -0.0885657
   Zero velocity time and angle for batch element 3: 1.40278, -0.0943745
   Zero velocity time and angle for batch element 0: 1.50592, -0.0798808
   Zero velocity time and angle for batch element 0: 2.50986, 0.0798808
   Zero velocity time and angle for batch element 1: 2.47168, 0.0836782
   Zero velocity time and angle for batch element 2: 2.43731, 0.0885657
   Zero velocity time and angle for batch element 3: 2.40688, 0.0943745
   Zero velocity time and angle for batch element 0: 3.51381, -0.0798808
   Zero velocity time and angle for batch element 1: 3.47567, -0.0836782
   Zero velocity time and angle for batch element 2: 3.44134, -0.0885657
   Zero velocity time and angle for batch element 3: 3.41099, -0.0943745
   Zero velocity time and angle for batch element 0: 4.51775, 0.0798808
   Zero velocity time and angle for batch element 1: 4.47965, 0.0836782
   Zero velocity time and angle for batch element 2: 4.44538, 0.0885657
   Zero velocity time and angle for batch element 3: 4.41509, 0.0943745

We can see how the event triggered 5 times for each batch element, and how
the oscillation period is roughly 1 for all batch elements. This is of course
expected due to the isochronicity property of the pendulum in the small oscillation
regime.

Full code listing
-----------------

.. literalinclude:: ../tutorial/batch_mode.cpp
   :language: c++
   :lines: 9-
