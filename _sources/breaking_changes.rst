.. _breaking_changes:

Breaking changes
================

.. _bchanges_4_0_0:

4.0.0
-----

heyoka 4 includes several backwards-incompatible changes.

API/behaviour changes
~~~~~~~~~~~~~~~~~~~~~

Step callbacks and ``propagate_*()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The way step callbacks interact with the ``propagate_*()`` functions has changed. Specifically:

- step callbacks are now passed by value into the ``propagate_*()`` functions (whereas previously
  they would be passed by reference), and
- step callbacks are now part of the return value. Specifically:

  - for the scalar ``propagate_for()`` and ``propagate_until()`` functions, the step callback is
    the sixth element of the return tuple, while for the batch variants the step callback
    is the second element of the return tuple;
  - for the scalar ``propagate_grid()`` function, the step callback is the fifth element of the return
    tuple, while for the batch variant the step callback is the first element of the return
    tuple.

:ref:`The ensemble propagation <tut_ensemble>` functions have been modified in an analogous way.

Adapting existing code for the new API should be straightforward. In most cases it should be just
a matter of:

- adapting strucuted bindings declarations to account for the new element in the return tuple
  of scalar propagations,
- adjusting the indexing into the return tuple when fetching a specific element,
- accounting for the fact that batch propagations now return a tuple of two elements
  rather than a single value.

``propagate_grid()``
^^^^^^^^^^^^^^^^^^^^

The ``propagate_grid()`` functions of the adaptive integrators now require the first element of the
time grid to be equal to the current integrator time. Previously, in case of a difference between the
integrator time and the first grid point, heyoka would propagate the state of the system up to the
first grid point with ``propagate_until()``.

If you want to recover the previous behaviour, you will have to invoke manually ``propagate_until(grid[0])``
before invoking ``propagate_grid()``.

General
~~~~~~~

- heyoka now requires LLVM>=13.
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
