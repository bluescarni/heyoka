.. _api_number:

Numerical constants
===================

.. cpp:namespace-push:: heyoka

*#include <heyoka/number.hpp>*

The :cpp:class:`~heyoka::number` class
--------------------------------------

.. cpp:class:: number

   This class is used to represent numerical constants in heyoka's expression
   system.

   It consists of a union of several floating-point types.

   .. cpp:type:: value_type = std::variant<float, double, long double, mppp::real128, mppp::real>

      .. note::

         :cpp:class:`mppp::real128` and :cpp:class:`mppp::real` are supported only if heyoka was built
         with support for the mp++ library (see the :ref:`installation instructions <installation>`).

      The union of floating-point types.

   .. cpp:function:: number() noexcept

      The default constructor initialises the internal union with the ``0.`` literal.

   .. cpp:function:: number(const number &)
   .. cpp:function:: number(number &&) noexcept
   .. cpp:function:: number &operator=(const number &)
   .. cpp:function:: number &operator=(number &&) noexcept
   .. cpp:function:: ~number()

      Numbers are copy/move constructible, copy/move assignable and destructible.

      Moved-from numbers are guaranteed to be in the default-constructed state.

      :exception: any exception thrown by the copy constructor/copy assignment operator of :cpp:class:`mppp::real`.

   .. cpp:function:: explicit number(float x) noexcept
   .. cpp:function:: explicit number(double x) noexcept
   .. cpp:function:: explicit number(long double x) noexcept
   .. cpp:function:: explicit number(mppp::real128 x) noexcept
   .. cpp:function:: explicit number(mppp::real x)

      .. note::

         The :cpp:class:`mppp::real128` and :cpp:class:`mppp::real` overloads are available only
         if heyoka was built with support for the mp++ library (see the :ref:`installation instructions <installation>`).

      Numbers can be initialised from values of the supported floating-point types.

      :param x: the construction argument.

      :exception: any exception raised by the copy constructor of :cpp:class:`mppp::real`.

   .. cpp:function:: [[nodiscard]] const value_type &value() const noexcept

      Value getter.

      :return: a reference to the internal union of floating-point types.

Functions
---------

.. cpp:function:: void swap(number &a, number &b) noexcept

   Swap primitive.

   This function will efficiently swap *a* and *b*.

   :param a: the first number.
   :param b: the second number.

.. cpp:function:: std::ostream &operator<<(std::ostream &os, const number &n)

   Stream operator.

   :param os: the output stream.
   :param n: the input number.

   :return: a reference to *os*.

   :exception: any exception thrown by streaming the value of *n*.

Operators
---------

.. cpp:function:: number operator+(number n)
.. cpp:function:: number operator-(const number &n)

   The :cpp:class:`~heyoka::number` class supports the identity and negation operators.

   :param n: the input argument.

   :return: *n* or its negation.

   :exception: any exception raised by the constructors of :cpp:class:`~heyoka::number`.

.. cpp:function:: number operator+(const number &x, const number &y)
.. cpp:function:: number operator-(const number &x, const number &y)
.. cpp:function:: number operator*(const number &x, const number &y)
.. cpp:function:: number operator/(const number &x, const number &y)

   The :cpp:class:`~heyoka::number` class supports elementary binary arithmetics.

   If the active floating-point types of *x* and *y* differ, the active type of the result
   will be the wider among the operands' types.

   :param x: the first operand.
   :param y: the second operand.

   :return: the result of the binary operation.

   :exception: any exception raised by the constructors of :cpp:class:`~heyoka::number` or by the implementation of the
    underlying arithmetic operation.
   :exception std\:\:invalid_argument: if the active types of *x* and *y* differ and they don't support mixed-mode airthmetics.

.. cpp:function:: bool operator==(const number &x, const number &y) noexcept
.. cpp:function:: bool operator!=(const number &x, const number &y) noexcept

   Equality comparison operators.

   Two numbers are considered equal if:

   - their active types are equal, and
   - their values are equal.

   Two NaN values are considered equivalent by these comparison operators.

   :param x: the first operand.
   :param y: the second operand.

   :return: the result of the comparison.

.. cpp:function:: bool operator<(const number &x, const number &y) noexcept

   Less-than comparison operator.

   *x* is less than *y* if:

   - the active type of *x* is narrower than the active type of *y*, or
   - the active types of *x* and *y* are the same, and the value of *x* is less than the value of *y*.

   NaN values are considered greater than non-NaN values by this operator.

   :param x: the first operand.
   :param y: the second operand.

   :return: the result of the comparison.

.. cpp:namespace-pop::

Standard library specialisations
--------------------------------

.. cpp:struct:: template <> std::hash<heyoka::number>

   Specialisation of ``std::hash`` for :cpp:class:`heyoka::number`.

   The hash value of NaNs depends only on the active floating-point type. That is, all NaNs
   of a floating-point type hash to the same value.

   .. cpp:function:: std::size_t operator()(const heyoka::number &n) const noexcept

      :param n: the input :cpp:class:`heyoka::number`.

      :return: a hash value for *n*.
