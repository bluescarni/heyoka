Changelog
=========

0.6.0 (unreleased)
------------------

New
~~~

- Start tracking code coverage
  (`#115 <https://github.com/bluescarni/heyoka/pull/115>`__).
- Initial version of the event detection system
  (`#107 <https://github.com/bluescarni/heyoka/pull/107>`__).
- Add a tutorial chapter for batch mode
  (`#106 <https://github.com/bluescarni/heyoka/pull/106>`__).
- Add a couple of utilities to detect the presence of the time
  function in an expression
  (`#105 <https://github.com/bluescarni/heyoka/pull/105>`__).
- Provide the ability to compute the jet of derivatives
  of arbitrary functions of the state variables
  (`#104 <https://github.com/bluescarni/heyoka/pull/104>`__).
- Speed-up the deep copy of just-in-time-compiled
  objects such as ``llvm_state`` and ``taylor_adaptive``
  (`#102 <https://github.com/bluescarni/heyoka/pull/102>`__).

Changes
~~~~~~~

- **BREAKING**: the ``propagate_grid()`` function now requires
  monotonically-ordered grid points
  (`#114 <https://github.com/bluescarni/heyoka/pull/114>`__).
- Change the screen output format for ``taylor_outcome``
  (`#106 <https://github.com/bluescarni/heyoka/pull/106>`__).

Fix
~~~

- In the batch integrator class, the outcomes in the result vectors
  are now initialised to ``taylor_outcome::success`` instead of
  meaningless values
  (`#102 <https://github.com/bluescarni/heyoka/pull/102>`__).

0.5.0 (2021-02-25)
------------------

New
~~~

- Implement various missing symbolic derivatives
  (`#101 <https://github.com/bluescarni/heyoka/pull/101>`__,
  `#100 <https://github.com/bluescarni/heyoka/pull/100>`__).
- Implement additional automatic simplifications
  in the expression system
  (`#100 <https://github.com/bluescarni/heyoka/pull/100>`__).
- Implement ``extract()`` for the ``func`` class, in order
  to retrieve a pointer to the type-erased inner object
  (`#100 <https://github.com/bluescarni/heyoka/pull/100>`__).

0.4.0 (2021-02-20)
------------------

New
~~~

- Introduce a dedicated negation operator in the
  expression system
  (`#99 <https://github.com/bluescarni/heyoka/pull/99>`__).
- Implement various new automatic simplifications
  in the expression system, and introduce ``powi()`` as
  an alternative exponentiation function for natural exponents
  (`#98 <https://github.com/bluescarni/heyoka/pull/98>`__).
- Implement propagation over a time grid
  (`#95 <https://github.com/bluescarni/heyoka/pull/95>`__).
- Implement support for dense output
  (`#92 <https://github.com/bluescarni/heyoka/pull/92>`__).
- Add the ability to output the Taylor coefficients
  when invoking the single-step functions in the
  integrator classes
  (`#91 <https://github.com/bluescarni/heyoka/pull/91>`__).

Fix
~~~

- Avoid division by zero in certain corner cases
  when using ``pow()`` with small natural exponents
  (`#98 <https://github.com/bluescarni/heyoka/pull/98>`__).

0.3.0 (2021-02-11)
------------------

New
~~~

- Implement the error function
  (`#89 <https://github.com/bluescarni/heyoka/pull/89>`__).
- Implement the standard logistic function
  (`#87 <https://github.com/bluescarni/heyoka/pull/87>`__).
- Implement the basic hyperbolic functions and their
  inverse counterparts
  (`#84 <https://github.com/bluescarni/heyoka/pull/84>`__).
- Implement the inverse trigonometric functions
  (`#81 <https://github.com/bluescarni/heyoka/pull/81>`__).
- The stream operator of functions can now be customised
  more extensively
  (`#78 <https://github.com/bluescarni/heyoka/pull/78>`__).
- Add explicit support for non-autonomous systems
  (`#77 <https://github.com/bluescarni/heyoka/pull/77>`__).
- heyoka now has a logo
  (`#73 <https://github.com/bluescarni/heyoka/pull/73>`__).

Changes
~~~~~~~

- Small optimisations in the automatic differentiation
  formulae
  (`#83 <https://github.com/bluescarni/heyoka/pull/83>`__).
- Improve common subexpression simplification in presence of
  constants of different types
  (`#82 <https://github.com/bluescarni/heyoka/pull/82>`__).
- Update copyright dates
  (`#79 <https://github.com/bluescarni/heyoka/pull/79>`__).
- Avoid using a temporary file when extracting the
  object code of a module
  (`#79 <https://github.com/bluescarni/heyoka/pull/79>`__).

Fix
~~~

- Ensure that ``pow(x ,0)`` always simplifies to 1,
  rather than producing an expression with null exponent
  (`#82 <https://github.com/bluescarni/heyoka/pull/82>`__).
- Fix build issue with older Boost versions
  (`#80 <https://github.com/bluescarni/heyoka/pull/80>`__).
- Various build system and doc fixes/improvements
  (`#88 <https://github.com/bluescarni/heyoka/pull/88>`__,
  `#86 <https://github.com/bluescarni/heyoka/pull/86>`__,
  `#85 <https://github.com/bluescarni/heyoka/pull/85>`__,
  `#83 <https://github.com/bluescarni/heyoka/pull/83>`__,
  `#82 <https://github.com/bluescarni/heyoka/pull/82>`__,
  `#76 <https://github.com/bluescarni/heyoka/pull/76>`__,
  `#74 <https://github.com/bluescarni/heyoka/pull/74>`__).

0.2.0 (2021-01-13)
------------------

New
~~~

- Extend the Taylor decomposition machinery to work
  on more general classes of functions, and add
  ``tan()``
  (`#71 <https://github.com/bluescarni/heyoka/pull/71>`__).
- Implement support for runtime parameters
  (`#68 <https://github.com/bluescarni/heyoka/pull/68>`__).
- Initial tutorials and various documentation additions
  (`#63 <https://github.com/bluescarni/heyoka/pull/63>`__).
- Add a stream operator for the ``taylor_outcome`` enum
  (`#63 <https://github.com/bluescarni/heyoka/pull/63>`__).

Changes
~~~~~~~

- heyoka now depends publicly on the Boost headers
  (`#68 <https://github.com/bluescarni/heyoka/pull/68>`__).

Fix
~~~

- Fix potential name mangling issues in compact mode
  (`#68 <https://github.com/bluescarni/heyoka/pull/68>`__).

0.1.0 (2020-12-18)
------------------

Initial release.
