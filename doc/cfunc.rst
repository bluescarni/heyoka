Compiled functions
==================

.. cpp:namespace-push:: heyoka

*#include <heyoka/expression.hpp>*

The :cpp:class:`cfunc` class
----------------------------

.. cpp:class:: template <typename T> cfunc

   Compiled function.

   .. cpp:function:: cfunc() noexcept 

      Default constructor.

      The default constructor inits the compiled function
      in an invalid state in which the only supported operations
      are destruction and assignment from a valid compiled
      function.
