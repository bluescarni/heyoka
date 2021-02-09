.. _tut_nonauto:

Non-autonomous systems
======================

All the ODE systems we have used in the examples thus far belong to the class of autonomous systems.
That is, the time variable :math:`t` never appears explicitly in the expressions of the ODEs. In this
section, we will see how non-autonomous systems can be defined and integrated in heyoka.

The dynamical system we will be focusing on is, again, a pendulum, but this time we will
spice things up a little by introducing a velocity-dependent damping effect and a time-dependent
external forcing. These additional effects create a rich and complex dynamical picture
which is highly sensitive to the initial conditions. See :cite:`hubbard1999forced`
for a detailed analysis of this dynamical system.

The ODE system of the forced damped pendulum reads:

.. math::

   \begin{cases}
   x^\prime = v \\
   v^\prime = \cos t - 0.1v - \sin(x)
   \end{cases}.

The :math:`\cos t` term represents a periodic time-dependent forcing, while :math:`-0.1v`
is a linear drag representing the effect of air on the pendulum's bob. Following :cite:`hubbard1999forced`,
we take as initial conditions

.. math::

   \begin{cases}
   x\left( 0 \right) = 0 \\
   v\left( 0 \right) = 1.97
   \end{cases}.

That is, the pendulum is initially in the vertical position with a non-zero velocity.

The time variable is represented in heyoka's expression system by a special placeholder
called, in a dizzying display of inventiveness, ``time``. Because the name ``time`` is fairly
common (e.g., ``time()`` is a function in the POSIX API), it is generally a good idea
to prepend the namespace ``heyoka`` (or its abbreviation, ``hy``) when using
the ``time`` expression, in order to avoid ambiguities that could confuse the compiler.
With that in mind, let's look at how the forced damped pendulum is defined in heyoka:

.. literalinclude:: ../tutorial/forced_damped_pendulum.cpp
   :language: c++
   :lines: 18-33

Note that, for the sake of completeness, we passed an explicit initial value for the time
variable via the keyword argument ``kw::time``. In this specific case, this is superfluous,
as the default initial value for the time variable is already zero.
