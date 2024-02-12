Compiled functions
==================

.. cpp:namespace-push:: heyoka

*#include <heyoka/expression.hpp>*

The :cpp:class:`cfunc` class
----------------------------

.. cpp:class:: template <typename T> cfunc

   Compiled function.

   This class allows to compile just-in-time symbolic functions
   defined via the :ref:`expression system <tut_expression_system>`.
   A :ref:`tutorial <tut_cfunc>` showcasing the use of this
   class is available.

   .. cpp:function:: cfunc() noexcept

      Default constructor.

      The default constructor inits the compiled function in an invalid
      state in which the only supported operations are copy/move
      construction/assignment, destruction and the invocation
      of the :cpp:func:`is_valid()` function.

   .. cpp:function:: template <typename... KwArgs> explicit cfunc(std::vector<expression> fn, std::vector<expression> vars, const KwArgs &...kw_args)

      Main constructor.

      This constructor will create a compiled function for the evaluation of the multivariate
      symbolic vector function *fn*. The *vars* argument is a list
      of :cpp:class:`variable` expressions representing the order in which the
      inputs of *fn* are passed to the call operator during evaluation.

   .. cpp:function:: [[nodiscard]] bool is_valid() const noexcept
