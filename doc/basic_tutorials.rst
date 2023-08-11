.. _basic_tutorials:

Basic tutorials
===============

.. important::

   More :ref:`tutorials <hypy:basic_tutorials>` and :ref:`examples <hypy:examples>` are available in the documentation
   of heyoka's `Python bindings <https://bluescarni.github.io/heyoka.py>`__.

The code snippets in these tutorials assume the inclusion of the
global header ``heyoka/heyoka.hpp``, and the use of

.. code-block:: c++

   using namespace heyoka;
   namespace hy = heyoka;

to import all names from the ``heyoka`` namespace, and to provide
a handy shortcut ``hy`` to the ``heyoka`` namespace.

The tutorials' code is available in the ``tutorials/`` subdirectory
of the source tree. The tutorials can be compiled by enabling the
``HEYOKA_BUILD_TUTORIALS`` option when
:ref:`compiling from source <installation_from_source>`.

.. toctree::
  :maxdepth: 1

  tut_taylor_method
  tut_expression_system
  tut_adaptive
  tut_adaptive_custom
  tut_param
  tut_nonauto
  tut_d_output
  tut_events
