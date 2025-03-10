.. _api_variable:

Variables
=========

.. cpp:namespace-push:: heyoka

*#include <heyoka/variable.hpp>*

The :cpp:class:`~heyoka::variable` class
----------------------------------------

.. cpp:class:: variable

   This class is used to represent symbolic variables in heyoka's expression
   system.

   Variables are uniquely identified by their name.

   .. note::

      Variable names beginning with a double underscore (``__``) are reserved
      for internal use.

   .. cpp:function:: variable() noexcept

      Default constructor.

      The default constructor initialises a variable with an empty name.

   .. cpp:function:: explicit variable(std::string name)

      Constructor from name.

      :param name: the name of the variable.

      :exception: any exception thrown by the copy constructor of ``std::string``.

   .. cpp:function:: variable(const variable &)
                     variable(variable &&) noexcept
                     variable &operator=(const variable &)
                     variable &operator=(variable &&) noexcept
                     ~variable()

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

Operators
---------

.. cpp:function:: bool operator==(const variable &a, const variable &b) noexcept
                  bool operator!=(const variable &a, const variable &b) noexcept

   Equality comparison operators.

   Two variables are considered equal if they have the same name.

   :param a: the first variable.
   :param b: the second variable.

   :return: the result of the comparison.

.. cpp:namespace-pop::

Standard library specialisations
--------------------------------

.. cpp:struct:: template <> std::hash<heyoka::variable>

   Specialisation of ``std::hash`` for :cpp:class:`heyoka::variable`.

   .. cpp:function:: std::size_t operator()(const heyoka::variable &v) const noexcept

      :param v: the input :cpp:class:`heyoka::variable`.

      :return: a hash value for *v*.
