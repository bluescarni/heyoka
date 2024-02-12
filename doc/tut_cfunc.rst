.. _tut_cfunc:

Compiled functions
==================

heyoka can compile just-in-time multivariate vector functions defined
via the :ref:`expression system <tut_expression_system>`. This feature
is described and explored in detail in a
:ref:`dedicated tutorial <hypy:cfunc_tut>` for heyoka.py, the
Python bindings of heyoka.

In Python, just-in-time compilation can lead to substantial
speedups for function evaluation. In C++, the performance argument
is less strong as C++ does not suffer from the same performance
pitfalls of Python: if you need fast evaluation of a function,
you can just implement it directly in C++ code without resorting
to just-in-time compilation.

Nevertheless, even in C++ heyoka's compiled functions offer
a few advantages over plain C++ functions:

- because functions are compiled just-in-time, they
  take advantage of all the features of the
  host CPU. Most importantly, heyoka's compiled functions
  support a batch evaluation mode which takes full advantage
  of `SIMD instructions <https://en.wikipedia.org/wiki/Single_instruction,_multiple_data>`__;
- batch mode evaluation of compiled functions also supports
  multithreaded parallelisation, which can provide a substantial
  performance boost on modern multicore machines;
- heyoka's functions support automatic differentiation up to
  arbitrary order, thus it is possible to evaluate the derivatives
  of a compiled function without any additional effort;
- heyoka's functions can be defined at runtime, whereas C++
  functions need to be defined and available at compilation time.
  This means that it is possible to create a compiled
  function at runtime from user-supplied data (e.g., a configuration
  file) and evaluate it with optimal performance.

The main downside of compiled functions is that the 
compilation process is computationally expensive and thus
just-in-time compilation is most useful when a function needs to
be evaluated repeatedly with different input values (so that the
initial compilation overhead can be absorbed by the evaluation
performance increase).

A simple example
----------------

As an initial example, we will JIT compile the simple bivariate
function

.. math::

   f\left(x, y \right) = x^2 - y^2.

We begin with the definition of the symbolic
variables and of the symbolic function to be compiled:

.. literalinclude:: ../tutorial/compiled_functions.cpp
   :language: c++
   :lines: 19-23

Next, we create a compiled function via the
:cpp:class:`~heyoka::cfunc` class:

.. literalinclude:: ../tutorial/compiled_functions.cpp
   :language: c++
   :lines: 25-26
