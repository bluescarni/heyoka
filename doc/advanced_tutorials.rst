.. _advanced_tutorials:

Advanced tutorials
==================

.. important::

  More :ref:`tutorials <hypy:tutorials>` and :ref:`examples <hypy:examples>` are available in the documentation
  of heyoka's `Python bindings <https://bluescarni.github.io/heyoka.py>`__.

In this section we will show some of heyoka's more advanced functionalities,
including multiprecision computations and vectorisation via batch mode.

Because in batch mode indexing over the state vector as a flat 1D array
can quickly become complicated and confusing, in these tutorials we will
make extensive use of `xtensor <https://xtensor.readthedocs.io/en/latest/>`__,
a C++ library which, among many other features, provides an API very similar
to `NumPy <https://numpy.org/>`__ for working with multidimensional arrays.
It is outside the scope of this document to give a full overview on xtensor's
API and capabilities. Here, however, we will use only some of xtensor's most basic
features, and, for a reader familiar with NumPy and its multidimensional array API,
the tutorials should not be too hard to follow.

.. toctree::
  :maxdepth: 1

  tut_batch_mode
  tut_s11n
  tut_ensemble
