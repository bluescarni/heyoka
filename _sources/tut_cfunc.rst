.. _tut_cfunc:

Compiled functions
==================

.. cpp:namespace-push:: heyoka

heyoka can compile just-in-time (JIT) multivariate vector functions defined
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
  can take advantage of all the features of the
  host CPU. Most importantly, heyoka's compiled functions
  support :ref:`batch evaluation <tut_cfunc_batch>`
  via `SIMD instructions <https://en.wikipedia.org/wiki/Single_instruction,_multiple_data>`__
  which can provide a multifold speed boost over
  plain (scalar) C++ functions;
- batch mode evaluation of compiled functions also supports
  multithreaded parallelisation, which can provide another substantial
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
:cpp:class:`cfunc` class:

.. literalinclude:: ../tutorial/compiled_functions.cpp
   :language: c++
   :lines: 25-26

Note how ``sym_func`` was passed to the constructor of
:cpp:class:`cfunc` enclosed in curly brackets:
this is because in general :cpp:class:`cfunc` expects
in input a vector function - that is, a list of expressions
representing the function components.
In this specific case, we are compiling a vector function with
only one component.

Like many other heyoka classes, :cpp:class:`cfunc` is
a class template parametrised over a single type ``T`` representing
the floating-point type to be used for function evaluation. In this
case, we are operating in standard ``double`` precision.

Let us inspect the compiled function object by printing
it to screen:

.. literalinclude:: ../tutorial/compiled_functions.cpp
   :language: c++
   :lines: 28-29

.. code-block:: console

   C++ datatype: double
   Variables: [x, y]
   Output #0: (x**2.0000000000000000 - y**2.0000000000000000)

We can now proceed to evaluate the compiled function. In order to do
so, we need to store the input values in a memory buffer and prepare
a memory buffer to store the result of the evaluation. We can use for
both ``std::array``:

.. literalinclude:: ../tutorial/compiled_functions.cpp
   :language: c++
   :lines: 31-33

We stored the values :math:`1` and :math:`2` in the input buffer, which
means that the function will be evaluated for :math:`x=1` and :math:`y=2`.

We can now proceed to invoke the call operator
of :cpp:class:`cfunc`, which will write the result of the
evaluation into ``out``:

.. literalinclude:: ../tutorial/compiled_functions.cpp
   :language: c++
   :lines: 35-36

Let us print ``out`` to screen in order to confirm that the evaluation
was successful:

.. literalinclude:: ../tutorial/compiled_functions.cpp
   :language: c++
   :lines: 38-39

.. code-block:: console

   Output: [-3]

.. _tut_cfunc_batch:

Batch evaluation
----------------

The simple example we have just seen consisted of the evaluation of a function
over a single value for each variable. :cpp:class:`cfunc` also supports
evaluation of a function over batches of input values for each variable.

In order to perform batch evaluation, we first have to define new
memory buffers to store the inputs and outputs of the evaluation.
We select a batch size of :math:`2`, which means we need storage for
:math:`2 \times 2 = 4` input values and :math:`2` output values:

.. literalinclude:: ../tutorial/compiled_functions.cpp
   :language: c++
   :lines: 41-43

In batch evaluations, input values for a single batch are expected to be stored
contiguously. That is, the input buffer will be interpreted as a row-major
bidimensional array in which each row contains the batch of input values for
a single variable. In this specific example, we will be evaluating the function
for :math:`x=\left[ 1, 1.1 \right]` and :math:`y=\left[ 2, 2.2 \right]`.

In the next step, we create bidimensional views over the input/output buffers
with the help of :cpp:type:`mdspan` and the convenience typedefs
:cpp:type:`cfunc::in_2d` and :cpp:type:`cfunc::out_2d`:

.. literalinclude:: ../tutorial/compiled_functions.cpp
   :language: c++
   :lines: 45-47

As we just explained, the input data is interpreted as a :math:`2 \times 2` array while
the input data is interpreted as a :math:`1 \times 2` array.

We are now ready to perform a batch evaluation:

.. literalinclude:: ../tutorial/compiled_functions.cpp
   :language: c++
   :lines: 49-50

Finally, we can print to screen the result of the evaluation:

.. literalinclude:: ../tutorial/compiled_functions.cpp
   :language: c++
   :lines: 52-53

.. code-block:: console

   Output: [-3, -3.6300000000000003]

For this simple example, we used a batch size of :math:`2`, but arbitrarily
large batch sizes are possible. If the batch size is large enough, heyoka
will parallelise the computation using multiple threads of execution, leading
to substantial speedups on multicore machines.

While in this tutorial we operated in standard ``double`` precision for simplicity,
compiled functions can also operate in single, extended, quadruple and multiple precision.
