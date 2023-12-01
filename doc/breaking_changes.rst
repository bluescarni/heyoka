.. _breaking_changes:

Breaking changes
================

.. _bchanges_4_0_0:

4.0.0
-----

heyoka 4 includes several backwards-incompatible changes.

General
~~~~~~~

- heyoka is now a C++20 project.
- heyoka now requires fmt>=9.
- heyoka now requires mp++ 1.x.

.. _bchanges_2_0_0:

2.0.0
-----

- The minimum supported LLVM version has been bumped
  from 10 to 11.

.. _bchanges_1_0_0:

1.0.0
-----

- The ``make_nbody_sys()`` function has been replaced by
  the ``model::nbody()`` function, with identical semantics.

.. _bchanges_0_16_0:

0.16.0
------

- The ``pairwise_sum()`` function has been replaced
  by a new function called ``sum()`` with similar semantics.
  ``sum()`` should behave as a drop-in replacement
  for ``pairwise_sum()``.
- The tuple returned by the ``propagate_for/until()`` functions
  in a scalar integrator has now 5 elements, rather than 4.
  The new return value at index 4 is the :ref:`continuous output <tut_c_output>`
  function object. This change can break code which assumes
  that the tuple returned by the ``propagate_for/until()`` functions
  has a size of 4, such as:

  .. code-block:: c++

     auto [r0, r1, r2, r3] = ta.propagate_until(...);

  The fix should be straightforward in most cases, e.g.:

  .. code-block:: c++

     auto [r0, r1, r2, r3, r4] = ta.propagate_until(...);

  Similarly, the ``propagate_for/until()`` functions in a batch integrator,
  which previously returned nothing, now return the :ref:`continuous output <tut_c_output>`
  function object.

.. _bchanges_0_15_0:

0.15.0
------

- The function class now uses reference
  semantics. This means that copy operations on
  non-trivial expressions now result in shallow copies,
  not deep copies (as it was previously the case).
  This change does not have repercussions on the
  integrators' API, but user code manipulating expressions
  may need to be adapted.

.. _bchanges_0_10_0:

0.10.0
------

- The callback that can (optionally) be passed to
  the ``propagate_*()`` functions must now return
  a ``bool`` indicating whether the integration should
  continue or not. The callback used to return ``void``.

.. _bchanges_0_8_0:

0.8.0
-----

- The direction of a non-terminal event is now specified
  with the (optional) keyword argument ``direction`` for
  the event's constructor (whereas before the direction
  could be specified via an unnamed argument).
- An ``int`` argument has been appended to the signature of
  the events' callbacks. This new argument represents the sign
  of the derivative of the event equation at the event trigger
  time, and its value will be -1 for negative derivative,
  1 for positive derivative and 0 for zero derivative.
