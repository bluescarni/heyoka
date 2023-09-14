.. _exceptions:

Exceptions
==========

*#include <heyoka/exceptions.hpp>*

.. cpp:namespace-push:: heyoka

.. cpp:class:: not_implemented_error final : public std::runtime_error

   Exception to signal that a feature/functionality is not implemented.

   This exception inherits all members (including constructors) from
   `std::runtime_error <https://en.cppreference.com/w/cpp/error/runtime_error>`_.
