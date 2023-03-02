.. _tut_nonauto:

Non-autonomous systems
======================

.. versionadded:: 0.3.0

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
   v\left( 0 \right) = 1.85
   \end{cases}.

That is, the pendulum is initially in the vertical position with a positive velocity.

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

We can now integrate the system for a few time units, checking how the value of :math:`x`
varies in time:

.. literalinclude:: ../tutorial/forced_damped_pendulum.cpp
   :language: c++
   :lines: 35-40

.. code-block:: console

   x = 3.49038
   x = 5.93825
   x = 7.30491
   x = 8.12543
   x = 5.12362
   x = 0.979573
   x = -0.90328
   x = -0.127736
   x = -0.773195
   x = 1.8008
   x = 2.71244
   x = -1.00752
   x = -1.55152
   x = 1.60996
   x = -0.880721
   x = -0.970923
   x = 2.35702
   x = 0.0993313
   x = -1.95449
   x = 1.46416
   x = 0.243313
   x = -1.949
   x = 1.55939
   x = 1.21015
   x = -2.06244

After an initial excursion to higher values for :math:`x`, the system seems to settle into a stable motion. Note that, because
this system can exhibit chaotic behaviour, changing
the initial conditions might lead to a qualitatively-different long-term behaviour.

Full code listing
-----------------

.. literalinclude:: ../tutorial/forced_damped_pendulum.cpp
   :language: c++
   :lines: 9-
