.. _expression_system:

Expression system
=================

.. cpp:namespace-push:: heyoka

The expression system is used to create and manipulate mathematical expressions in symbolic form.
An :cpp:class:`expression` is a union of several types:

- :ref:`symbolic variables <api_variable>`,
- :ref:`numerical constants <api_number>`,
- :ref:`runtime parameters <api_param>`,
- :ref:`n-ary functions <api_func>`.

Arithmetic operators and several :ref:`mathematical functions <api_math>` can be used
to construct arbitrarily-complicated symbolic expressions.

A :ref:`tutorial <tut_expression_system>` showcasing the capabilities of the expression
system is available.

.. toctree::
   :maxdepth: 1

   variable.rst
   number.rst
   param.rst
   func.rst
   expression.rst
   math.rst
