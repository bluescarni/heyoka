.. _tut_c_output:

Continuous output
=================

.. versionadded:: 0.4.0

One of the peculiar features of Taylor's method is that it directly provides,
via the Taylor series :eq:`tts_01`, *continuous* (or *dense*) output.
That is, the Taylor series built by the integrator at each timestep can be used
to compute the solution of the ODE system at *any time* within the timestep
(and not only at the endpoint).

Because the construction of the Taylor series is part of the timestepping algorithm,
support for continuous output comes at essentially no extra
cost. Additionally, because the continuous output is computed via the
Taylor series of the solution of the ODE system, its accuracy
is guaranteed to match the error tolerance of the integrator.
