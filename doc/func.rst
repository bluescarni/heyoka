Functions
=========

*#include <heyoka/func.hpp>*

The :cpp:class:`~heyoka::func_base` class
-----------------------------------------

.. cpp:namespace-push:: heyoka

.. cpp:class:: func_base

   This is the base class of all functions implemented in the expression system.

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
