Expressions
===========

.. cpp:namespace-push:: heyoka

*#include <heyoka/expression.hpp>*

The :cpp:class:`expression` class
---------------------------------

.. cpp:class:: expression

Functions
---------

.. cpp:function:: template <typename Arg0, typename... Args> auto make_vars(const Arg0 &str, const Args &...strs)

   Create variable expressions from strings.

   This function will return one or more :cpp:class:`expression` instances
   containing :cpp:class:`variables <variable>` constructed from the input arguments.
   If a single argument is supplied, a single expression is returned. Otherwise, a ``std::array`` of
   expressions (one for each argument) is returned.

   This function is enabled only if all input arguments are convertible to ``std::string``.

   :param str: the first string argument.
   :param strs: the remaining string arguments.

   :return: one or more expressions constructed from *str* and *strs*.

   :exception: any exception thrown by constructing ``std::string`` objects.
