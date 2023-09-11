Macros and definitions
======================

*#include <heyoka/config.hpp>*

.. c:macro:: HEYOKA_VERSION_STRING

   This definition expands to a string literal containing the full version of the heyoka library
   (e.g., for version 1.2.3 this macro expands to ``"1.2.3"``).

.. c:macro:: HEYOKA_VERSION_MAJOR

   This definition expands to an integral literal corresponding to heyoka's major version number (e.g.,
   for version 1.2.3, this macro expands to ``1``).

.. c:macro:: HEYOKA_VERSION_MINOR

   This definition expands to an integral literal corresponding to heyoka's minor version number (e.g.,
   for version 1.2.3, this macro expands to ``2``).

.. c:macro:: HEYOKA_VERSION_PATCH

   This definition expands to an integral literal corresponding to heyoka's patch version number (e.g.,
   for version 1.2.3, this macro expands to ``3``).

.. c:macro:: HEYOKA_WITH_MPPP

   This name is defined if heyoka was built with support for the mp++ library (see the :ref:`installation instructions <installation>`).

.. c:macro:: HEYOKA_WITH_REAL128
.. c:macro:: HEYOKA_WITH_REAL

    These names are defined if heyoka was built with support for, respectively, the :cpp:class:`~mppp::real128` and :cpp:class:`~mppp::real` classes from
    the mp++ library (see the :ref:`installation instructions <installation>`).

.. c:macro:: HEYOKA_WITH_SLEEF

   This name is defined if heyoka was built with support for the SLEEF library (see the :ref:`installation instructions <installation>`).
