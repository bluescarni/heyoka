Expressions
===========

.. cpp:namespace-push:: heyoka

*#include <heyoka/expression.hpp>*

The :cpp:class:`expression` class
---------------------------------

.. cpp:class:: expression

   Class to represent symbolic expressions.

   This is the main class used to represent mathematical expressions in heyoka.
   It is a union of several types:

   - :ref:`symbolic variables <api_variable>`,
   - :ref:`numerical constants <api_number>`,
   - :ref:`runtime parameters <api_param>`,
   - :ref:`n-ary functions <api_func>`.

   Because expressions are essentially `trees <https://en.wikipedia.org/wiki/Tree_(data_structure)>`__,
   we refer to these types as the *node types* of an expression.

   Expressions can be created in a variety of ways. After creation, expressions
   can be combined via :ref:`arithmetic operators <api_ex_ops>` and :ref:`mathematical functions <api_math>`
   to form new expressions of arbitrary complexity.

   Expressions which consist of a single :ref:`variable <api_variable>` or a single
   :ref:`constant <api_number>`/:ref:`parameter <api_param>` are referred to as
   *elementary* expressions.

   Expressions provide an immutable API: after creation, an expression cannot be changed in-place
   (except via assignment or swapping).

   .. cpp:type:: value_type = std::variant<number, variable, func, param>

      The union of node types.

   .. cpp:function:: expression() noexcept

      Default constructor.

      This constructor initialises the expression to a double-precision :ref:`number <api_number>` with a value of zero.

   .. cpp:function:: explicit expression(float x) noexcept

   .. cpp:function:: explicit expression(double x) noexcept

   .. cpp:function:: explicit expression(long double x) noexcept

   .. cpp:function:: explicit expression(mppp::real128 x) noexcept

   .. cpp:function:: explicit expression(mppp::real x)

      Constructors from floating-point objects.

      These constructors initialise the expression to a floating-point :ref:`number <api_number>` with the input value *x*.
      Expressions can be constructed from objects of any floating-point type supported by :ref:`number <api_number>`.

      :param x: the construction argument.

      :exception: any exception raised by the copy constructor of :cpp:class:`mppp::real`.

   .. cpp:function:: explicit expression(std::string s)

      Constructor from variable name.

      This constructor initialises the expression to a :ref:`variable <api_variable>` constructed from the
      input string *s*.

      :param s: the variable name.

      :exception: any exception thrown by the copy constructor of ``std::string``.

   .. cpp:function:: explicit expression(number x)

   .. cpp:function:: explicit expression(variable x)

   .. cpp:function:: explicit expression(func x) noexcept

   .. cpp:function:: explicit expression(param x) noexcept

      Constructors from objects of the node types.

      These constructors will initialise the internal union with the input argument *x*.

      :param x: the construction argument.

      :exception: any exception raised by the copy constructor of :cpp:class:`number` or :cpp:class:`variable`.

   .. cpp:function:: expression(const expression &)

   .. cpp:function:: expression(expression &&) noexcept

   .. cpp:function:: expression &operator=(const expression &)

   .. cpp:function:: expression &operator=(expression &&) noexcept

   .. cpp:function:: ~expression()

      Expressions are copy/move constructible/assignable and destructible.

      Note that because :cpp:class:`func` employs reference semantics, copying/assigning
      a non-elementary expression is a constant-time operation.

      :exception: any exception thrown by the copy constructor/copy assignment operators of the active node types.

   .. cpp:function:: [[nodiscard]] const value_type &value() const noexcept

      Const accessor to the internal union.

      :return: a const reference to the internal :cpp:type:`value_type` instance.

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

   Example
   ~~~~~~~

   .. code-block:: c++

      auto x = make_vars("x");
      auto [y, z] = make_vars("y", "z");

.. _api_ex_ops:

Operators
---------

Arithmetic operators
^^^^^^^^^^^^^^^^^^^^

The :cpp:class:`expression` class provides overloaded arithmetic binary operators and their in-place variants.

The overloaded binary operators require at least one argument to be an :cpp:class:`expression`, while
the other argument can be
either another :cpp:class:`expression` or any floating-point value supported by :cpp:class:`number`.

The overloaded in-place operators require the first argument to be an :cpp:class:`expression`, while
the second argument can be
either another :cpp:class:`expression` or any floating-point value supported by :cpp:class:`number`.

Comparison operators
^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: bool operator==(const expression &e1, const expression &e2) noexcept

.. cpp:function:: bool operator!=(const expression &e1, const expression &e2) noexcept

   Expression (in)equality.

   These operators compare *e1* and *e2* for **structural** equality. That is, two expressions are considered
   equal if the underlying symbolic trees are identical. It is important to emphasise that while structural
   equality implies mathematical equivalence, the opposite is not true: it is possible to define
   structurally-different expressions which are mathematically equivalent, such as
   :math:`\sin^2\left(x\right)+\cos^2\left(x\right)` and :math:`1`.

   :param e1: the first operand.
   :param e2: the second operand.

   :return: the result of the comparison.

User-defined literals
---------------------

.. cpp:function:: expression literals::operator""_flt(long double)

.. cpp:function:: expression literals::operator""_flt(unsigned long long)

.. cpp:function:: expression literals::operator""_dbl(long double)

.. cpp:function:: expression literals::operator""_dbl(unsigned long long)

.. cpp:function:: expression literals::operator""_ldbl(long double)

.. cpp:function:: expression literals::operator""_ldbl(unsigned long long)

.. cpp:function:: template <char... Chars> expression literals::operator""_f128()
