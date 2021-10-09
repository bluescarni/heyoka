.. _breaking_changes:

Breaking changes
================

0.16.0
------

- The ``pairwise_sum()`` function has been replaced
  by a new function called ``sum()`` with similar semantics.
  ``sum()`` should behave in most cases as a drop-in replacement
  for ``pairwise_sum()``.

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
