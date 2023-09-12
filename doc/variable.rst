.. _variable:

The :cpp:class:`~heyoka::variable` class
========================================

*#include <heyoka/variable.hpp>*

.. cpp:namespace-push:: heyoka

.. cpp:class:: variable

   This class is used to represent symbolic variables in heyoka's expression
   system. Variables are uniquely identified by their name.

   .. note::

      Variable names beginning with a double underscore (``__``) are reserved
      for internal use.

   .. cpp:function:: variable()

      Default constructor.

      The default constructor initialises a variable with an empty name.

   .. cpp:function:: explicit variable(std::string name)

      Constructor from name.

      :param name: the name of the variable.

      :exception: any exception thrown by the copy constructor of ``std::string``.

   .. cpp:function:: variable(const variable &)
   .. cpp:function:: variable(variable &&) noexcept
   .. cpp:function:: variable &operator=(const variable &)
   .. cpp:function:: variable &operator=(variable &&) noexcept
   .. cpp:function:: ~variable()

      Variables are copy/move constructible, copy/move assignable and destructible.

      :exception: any exception thrown by the copy constructor/copy assignment operator of ``std::string``.

   .. cpp:function:: [[nodiscard]] const std::string &name() const noexcept

      Name getter.

      :return: a reference to the name of the variable.

Functions
---------

.. cpp:function:: void swap(variable &a, variable &b) noexcept

   Swap primitive.

   This function will efficiently swap *a* and *b*.

   :param a: the first variable.
   :param b: the second variable.

.. cpp:function:: std::ostream &operator<<(std::ostream &os, const variable &v)

   Stream operator.

   :param os: the output stream.
   :param v: the input variable.

   :return: a reference to *os*.

   :exception: any exception thrown by the stream operator of ``std::string``.

.. cpp:function:: bool operator==(const variable &a, const variable &b) noexcept
.. cpp:function:: bool operator!=(const variable &a, const variable &b) noexcept

   Equality comparison operators.

   Two variables are considered equal if they have the same name.

   :param a: the first variable.
   :param b: the second variable.

   :return: the result of the comparison.
