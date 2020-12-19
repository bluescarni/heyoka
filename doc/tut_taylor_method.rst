.. _tut_taylor_method:

Taylor's method
===============

Taylor's method for solving systems of ordinary differential equations
(ODEs) is conceptually simple. Given an ODE system in the general
explicit from

.. math::
   :label: ode_sys_00

   \frac{d \boldsymbol{x}\left( t \right)}{dt}=\boldsymbol{F}\left(t, \boldsymbol{x}\left( t \right) \right),

with initial conditions at :math:`t=t_0`

.. math::
   :label: ic_00

   \boldsymbol{x}_0=\boldsymbol{x}\left(t_0\right),

Taylor's method approximates the value of the solution at :math:`t=t_1` as the truncated Taylor series
expansion of :math:`\boldsymbol{x}\left( t \right)` around :math:`t=t_0`:

.. math::
   :label: tts_00

   \boldsymbol{x}\left( t_1 \right) = \boldsymbol{x}_0 + \boldsymbol{x}'\left(t_0\right)h 
   +\frac{1}{2}\boldsymbol{x}''\left(t_0\right)h^2+\ldots+\frac{\boldsymbol{x}^{\left( p \right)}\left(t_0\right)}{p!}h^p,

where :math:`h=t_1-t_0` is the integration timestep and :math:`p` is the order of the Taylor method. Eq. :eq:`tts_00`
can be rewritten in more compact form as

.. math::
   :label: tts_01

   \boldsymbol{x}\left( t_1 \right) = \sum_{n=0}^p \boldsymbol{x}^{\left[ n \right]} \left(t_0\right) h^n,

where we have defined

.. math::
   :label: norm_d_00

   \boldsymbol{x}^{\left[ n \right]}\left( t \right) = \frac{1}{n!}\boldsymbol{x}^{\left( n \right)}\left( t \right)

as the normalised derivative of :math:`\boldsymbol{x}\left( t \right)` of order :math:`n`.

The derivatives appearing in the Taylor polynomial :eq:`tts_01` can be computed from the
initial conditions :eq:`ic_00` and the right-hand side of the ODE system :eq:`ode_sys_00`.
Thus, :eq:`tts_01` effectively can be used as a time stepper
whose precision and performance characteristics depend on the choice of the step size :math:`h` and
the Taylor order :math:`p`. 

Taylor integrators need to be... tailored to the specific expression of :math:`\boldsymbol{F}\left(t, \boldsymbol{x}\left( t \right) \right)`.
That is, unlike other popular numerical integration methods (e.g., `Runge-Kutta methods <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`__),
Taylor's method requires the user not only to provide :math:`\boldsymbol{F}\left(t, \boldsymbol{x}\left( t \right) \right)`,
but also to implement all the derivatives necessary to construct the Taylor polynomial :eq:`tts_01`.
This task, if done by hand, can be extremely cumbersome, inefficient and error-prone, especially for large ODE
systems and/or high-accuracy applications.

The main functionality provided by heyoka is the ability to automatically synthesise a complete Taylor integrator starting
only from a
symbolic representation of :math:`\boldsymbol{F}\left(t, \boldsymbol{x}\left( t \right) \right)` and a set of initial
conditions. Specifically,
given a differentiable symbolic expression for :math:`\boldsymbol{F}\left(t, \boldsymbol{x}\left( t \right) \right)`,
heyoka takes care of:

* the computation of the high-order derivatives necessary to implement the time stepper :eq:`tts_01`
  via a process of automatic differentiation,
* the deduction of optimal values for the Taylor order :math:`p` and the (adaptive) step size :math:`h`,
* the propagation of the state of the system via the evaluation of the Taylor polynomial :eq:`tts_01`.

In order to represent symbolically :math:`\boldsymbol{F}\left(t, \boldsymbol{x}\left( t \right) \right)`, heyoka
relies on a small, self-contained symbolic expression system (similar to an extremely trimmed-down,
bare-bones `computer algebra system <https://en.wikipedia.org/wiki/Computer_algebra_system>`__).
The expression system is used to decompose :math:`\boldsymbol{F}\left(t, \boldsymbol{x}\left( t \right) \right)`
into a sequence of elementary subexpression on which automatic differentiation rules are applied.
The sequence of operations necessary to compute the high-order derivatives of
:math:`\boldsymbol{F}\left(t, \boldsymbol{x}\left( t \right) \right)`
is then assembled and compiled just-in-time (via `LLVM <https://llvm.org/>`__) to
produce a time stepper function usable from regular C++ code.
