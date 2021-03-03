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
