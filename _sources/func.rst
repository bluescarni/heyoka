.. _api_func:

N-ary functions
===============

.. cpp:namespace-push:: heyoka

*#include <heyoka/func.hpp>*

The :cpp:class:`~heyoka::func_base` class
-----------------------------------------

.. cpp:class:: func_base

   This is the base class of all functions implemented in the :ref:`expression system<expression_system>`.

   It provides a common interface for constructing functions and accessing their arguments.

   .. cpp:function:: explicit func_base(std::string name, std::vector<expression> args)

      Constructor from name and arguments.

      :param name: the function name.
      :param args: the function arguments.

      :exception: any exceptions raised by copying the name or the arguments.
      :exception std\:\:invalid_argument: if *name* is empty.

   .. cpp:function:: func_base(const func_base &)
   .. cpp:function:: func_base(func_base &&) noexcept
   .. cpp:function:: func_base &operator=(const func_base &)
   .. cpp:function:: func_base &operator=(func_base &&) noexcept
   .. cpp:function:: ~func_base()

      :cpp:class:`func_base` is copy/move constructible, copy/move assignable and destructible.

      :exception: any exception thrown by the copy constructor/copy assignment operator the function's name or arguments.

   .. cpp:function:: [[nodiscard]] const std::string &get_name() const noexcept

      Name getter.

      :return: a reference to the function's name.

   .. cpp:function:: [[nodiscard]] const std::vector<expression> &args() const noexcept

      Arguments getter.

      :return: a reference to the function's arguments.

The :cpp:class:`~heyoka::func` class
------------------------------------

.. cpp:class:: func

   This class is used to represent functions in the :ref:`expression system<expression_system>`.

   :cpp:class:`func` is a polymorphic wrapper that can be constructed from any
   object satisfying certain conceptual requirements. We refer to such objects as
   *user-defined functions*, or UDFs (see the :cpp:concept:`is_udf` concept).

   The polymorphic wrapper is implemented internally via ``std::shared_ptr``, and thus
   :cpp:class:`func` employs reference semantics: copy construction and copy assignment
   do not throw, and they are constant-time operations.

   .. note::

      At this time, the :cpp:class:`func` API is still in flux and as such
      it is largely undocumented. Please refer to the source
      code if you need to understand the full API.

   .. cpp:function:: func()

      Default constructor.

      The default constructor will initialise this :cpp:class:`func` object
      with an implementation-defined UDF.

      :exception: any exception thrown by memory allocation failures.

   .. cpp:function:: template <typename T> requires (!std::same_as<func, std::remove_cvref_t<T>>) && is_udf<std::remove_cvref_t<T>> explicit func(T &&x)

      Generic constructor.

      This constructor will initialise ``this`` with the user-defined function *x*.

      :exception: any exception thrown by memory allocation failures or by the copy/move constructor of the user-defined function *x*.

   .. cpp:function:: func(const func &) noexcept
   .. cpp:function:: func(func &&) noexcept
   .. cpp:function:: func &operator=(const func &) noexcept
   .. cpp:function:: func &operator=(func &&) noexcept
   .. cpp:function:: ~func()

      :cpp:class:`func` is copy/move constructible, copy/move assignable and destructible.

      The only valid operations on a moved-from :cpp:class:`func` are destruction and copy/move assignment.

   .. cpp:function:: [[nodiscard]] const std::string &get_name() const noexcept

      Name getter.

      This getter will invoke :cpp:func:`func_base::get_name()` on the internal UDF.

      :return: a reference to the function's name.

   .. cpp:function:: [[nodiscard]] const std::vector<expression> &args() const noexcept

      Arguments getter.

      This getter will invoke :cpp:func:`func_base::args()` on the internal UDF.

      :return: a reference to the function's arguments.

Concepts
--------

.. cpp:concept:: template <typename T> is_udf = std::default_initializable<T> && std::copyable<T> && std::derived_from<T, func_base>

   User-defined function concept.

   This concept enumerates the minimum requirements of user-defined functions (UDFs), that is, objects that
   can be used to construct a :cpp:class:`func`.
