Changelog
=========

0.15.0 (unreleased)
-------------------

New
~~~

- Add an implementation of the VSOP2013 analytical solution
  for the motion of the planets of the Solar System, usable
  in the definition of differential equations
  (`#183 <https://github.com/bluescarni/heyoka/pull/183>`__,
  `#180 <https://github.com/bluescarni/heyoka/pull/180>`__).
- Add the two-argument inverse tangent function ``atan2()``
  to the expression system
  (`#182 <https://github.com/bluescarni/heyoka/pull/182>`__).
- Implement additional automatic simplifications for sin/cos
  (`#179 <https://github.com/bluescarni/heyoka/pull/179>`__).

Fix
~~~

- Fix an issue in ``kepE()`` arising from an automatic simplification
  that would lead to an invalid decomposition for zero eccentricity
  (`#185 <https://github.com/bluescarni/heyoka/pull/185>`__).

0.14.0 (2021-08-03)
-------------------

New
~~~

- The tolerance value is now stored in the integrator objects
  (`#175 <https://github.com/bluescarni/heyoka/pull/175>`__).

Changes
~~~~~~~

- Improve the heuristic for the automatic deduction
  of the cooldown value for terminal events
  (`#178 <https://github.com/bluescarni/heyoka/pull/178>`__).

Fix
~~~

- Ensure that code generation in compact mode is platform-agnostic
  and deterministic across executions
  (`#176 <https://github.com/bluescarni/heyoka/pull/176>`__).

0.12.0 (2021-07-21)
-------------------

New
~~~

- Add support for 64-bit PowerPC processors
  (`#171 <https://github.com/bluescarni/heyoka/pull/171>`__).
- Add support for 64-bit ARM processors
  (`#167 <https://github.com/bluescarni/heyoka/pull/167>`__).
- Implement serialisation for the main classes via
  Boost.Serialization
  (`#163 <https://github.com/bluescarni/heyoka/pull/163>`__).

Fix
~~~

- Fix a bug in the move assignment operator of ``llvm_state``
  (`#163 <https://github.com/bluescarni/heyoka/pull/163>`__).

0.11.0 (2021-07-06)
-------------------

New
~~~

- The ``time`` expression now supports symbolic
  differentiation
  (`#160 <https://github.com/bluescarni/heyoka/pull/160>`__).

Changes
~~~~~~~

- Various performance optimisations for the creation
  of large ODE systems
  (`#152 <https://github.com/bluescarni/heyoka/pull/152>`__).

0.10.1 (2021-07-02)
-------------------

Fix
~~~

- Parameters in event equations are now correctly counted
  when inferring the total number of parameters in an ODE system
  (`#154 <https://github.com/bluescarni/heyoka/pull/154>`__).

0.10.0 (2021-06-09)
-------------------

New
~~~

- The callback that can be passed to the ``propagate_*()`` functions
  can now be used to stop the integration
  (`#149 <https://github.com/bluescarni/heyoka/pull/149>`__).
- Add a pairwise product primitive
  (`#147 <https://github.com/bluescarni/heyoka/pull/147>`__).

Changes
~~~~~~~

- **BREAKING**: a :ref:`breaking change <bchanges_0_10_0>`
  in the ``propagate_*()`` callback API
  (`#149 <https://github.com/bluescarni/heyoka/pull/149>`__).
- Implement additional automatic simplifications in the expression system
  (`#148 <https://github.com/bluescarni/heyoka/pull/148>`__).
- Division by zero in the expression system now raises an error
  (`#148 <https://github.com/bluescarni/heyoka/pull/148>`__).

0.9.0 (2021-05-25)
------------------

New
~~~

- Add time polynomials to the expression system
  (`#144 <https://github.com/bluescarni/heyoka/pull/144>`__).
- Add the inverse of Kepler's elliptic equation to the expression
  system
  (`#138 <https://github.com/bluescarni/heyoka/pull/138>`__).
- Add an LLVM-based vectorised solver for Kepler's equation
  (`#136 <https://github.com/bluescarni/heyoka/pull/136>`__).
- Add an LLVM ``while`` loop function
  (`#135 <https://github.com/bluescarni/heyoka/pull/135>`__).

Changes
~~~~~~~

- Performance improvements for event detection in the linear
  and quadratic cases
  (`#145 <https://github.com/bluescarni/heyoka/pull/145>`__).
- Several functions used for event detection are now
  compiled just-in-time, rather than being implemented
  in C++
  (`#142 <https://github.com/bluescarni/heyoka/pull/142>`__).
- Cleanup unused and undocumented functions
  (`#134 <https://github.com/bluescarni/heyoka/pull/134>`__).
- Small performance optimisations
  (`#133 <https://github.com/bluescarni/heyoka/pull/133>`__).
- Remove the ``binary_operator`` node type in the expression
  system and implement binary arithmetic using the ``func`` node
  type instead
  (`#132 <https://github.com/bluescarni/heyoka/pull/132>`__). This
  is an internal change that does not affect the integrators' API.

0.8.0 (2021-04-28)
------------------

New
~~~

- The ``propagate_for/until()`` functions now support writing
  the Taylor coefficients at the end of each timestep
  (`#131 <https://github.com/bluescarni/heyoka/pull/131>`__).

Changes
~~~~~~~

- **BREAKING**: various :ref:`breaking changes <bchanges_0_8_0>`
  in the event detection API
  (`#131 <https://github.com/bluescarni/heyoka/pull/131>`__).
- Improvements to the stream operator of ``taylor_outcome``
  (`#131 <https://github.com/bluescarni/heyoka/pull/131>`__).

Fix
~~~

- Don't set the multiroot ``mr`` flag to ``true`` if
  a terminal event has a cooldown of zero
  (`#131 <https://github.com/bluescarni/heyoka/pull/131>`__).

0.7.0 (2021-04-21)
------------------

New
~~~

- Support LLVM 12
  (`#128 <https://github.com/bluescarni/heyoka/pull/128>`__).
- The ``propagate_*()`` functions now accept an optional
  ``max_delta_t`` argument to limit the size of a timestep,
  and an optional ``callback`` argument that will be invoked
  at the end of each timestep
  (`#127 <https://github.com/bluescarni/heyoka/pull/127>`__).
- The time coordinate in the Taylor integrator classes
  is now represented internally in double-length format. This change
  greatly reduces the error in long-term integrations of
  non-autonomous systems and improves the time accuracy
  of the predicted state
  (`#126 <https://github.com/bluescarni/heyoka/pull/126>`__).
- ``update_d_output()`` can now be called with a relative
  (rather than absolute) time argument
  (`#126 <https://github.com/bluescarni/heyoka/pull/126>`__).

Changes
~~~~~~~

- Performance improvements for the event detection system
  (`#129 <https://github.com/bluescarni/heyoka/pull/129>`__).
- **BREAKING**: the time coordinates in batch integrators
  cannot be directly modified any more, and the new
  ``set_time()`` function must be used instead
  (`#126 <https://github.com/bluescarni/heyoka/pull/126>`__).

Fix
~~~

- Fix an issue in the automatic deduction of the cooldown time
  for terminal events
  (`#126 <https://github.com/bluescarni/heyoka/pull/126>`__).

0.6.1 (2021-04-08)
------------------

Changes
~~~~~~~

- The event equations are now taken into account in the
  determination of the adaptive timestep
  (`#124 <https://github.com/bluescarni/heyoka/pull/124>`__).

Fix
~~~

- Fix an initialisation order issue in the event detection code
  (`#124 <https://github.com/bluescarni/heyoka/pull/124>`__).
- Fix an assertion misfiring in the event detection function
  (`#123 <https://github.com/bluescarni/heyoka/pull/123>`__).

0.6.0 (2021-04-06)
------------------

New
~~~

- Implement ``propagate_grid()`` for the batch integrator
  (`#119 <https://github.com/bluescarni/heyoka/pull/119>`__).
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
