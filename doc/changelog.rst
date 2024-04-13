Changelog
=========

4.1.0 (unreleased)
------------------

New
~~~

- Add mutable ranges getters for the state and pars data of the adaptive
  integrators (`#409 <https://github.com/bluescarni/heyoka/pull/409>`__).
- Support LLVM 18 (`#408 <https://github.com/bluescarni/heyoka/pull/408>`__).

Changes
~~~~~~~

- Remove the (undocumented) ``taylor_add_jet()`` function and rework
  the unit test code to use ``taylor_adaptive`` instead
  (`#409 <https://github.com/bluescarni/heyoka/pull/409>`__).

Fix
~~~

- Fix test failures on OSX arm64
  (`#409 <https://github.com/bluescarni/heyoka/pull/409>`__).

4.0.3 (2024-04-04)
------------------

Fix
~~~

- Workaround compilation failures in the unit tests
  when using GCC 13
  (`#409 <https://github.com/bluescarni/heyoka/pull/409>`__).
- Fix compilation on FreeBSD
  (`#407 <https://github.com/bluescarni/heyoka/pull/407>`__).

4.0.2 (2024-03-03)
------------------

Fix
~~~

- Fix compilation on MinGW
  (`#404 <https://github.com/bluescarni/heyoka/pull/404>`__).

4.0.1 (2024-03-02)
------------------

Fix
~~~

- Fix compilation on PowerPC
  (`#401 <https://github.com/bluescarni/heyoka/pull/401>`__).

4.0.0 (2024-03-02)
------------------

New
~~~

- heyoka is now available in the `spack <https://github.com/spack/spack>`__
  package manager.
- New :ref:`tutorial <tut_cfunc>` on compiled functions
  (`#396 <https://github.com/bluescarni/heyoka/pull/396>`__).
- New :cpp:class:`~heyoka::cfunc` class to facilitate
  the creation and evaluation of compiled functions, supporting
  automatic multithreaded parallelisation
  (`#396 <https://github.com/bluescarni/heyoka/pull/396>`__).
- It is now possible to index into the tensors of derivatives
  using indices vectors in sparse format
  (`#389 <https://github.com/bluescarni/heyoka/pull/389>`__).
- Add support for Lagrangian and Hamiltonian mechanics
  (`#381 <https://github.com/bluescarni/heyoka/pull/381>`__,
  `#379 <https://github.com/bluescarni/heyoka/pull/379>`__).
- It is now possible to pass a range of step callbacks to the
  ``propagate_*()`` functions. The individual callbacks will be
  automatically composed into a callback set
  (`#376 <https://github.com/bluescarni/heyoka/pull/376>`__).
- New ``angle_reducer`` step callback to automatically reduce
  angular state variables to the :math:`\left[0, 2\pi\right)` range
  (`#376 <https://github.com/bluescarni/heyoka/pull/376>`__).
- New ``callback`` module containing ready-made step and event callbacks
  (`#376 <https://github.com/bluescarni/heyoka/pull/376>`__).

Changes
~~~~~~~

- Speedups for the ``subs()`` primitive
  (`#394 <https://github.com/bluescarni/heyoka/pull/394>`__).
- **BREAKING**: the :cpp:func:`~heyoka::make_vars()` function
  now returns a single expression (rather than an array of expressions)
  if a single argument is passed in input
  (`#386 <https://github.com/bluescarni/heyoka/pull/386>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- **BREAKING**: the signature of callbacks for terminal events
  has been simplified
  (`#385 <https://github.com/bluescarni/heyoka/pull/385>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- **BREAKING**: the way in which the ``propagate_*()`` functions
  interact with step callbacks has changed
  (`#376 <https://github.com/bluescarni/heyoka/pull/376>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- **BREAKING**: the ``propagate_grid()`` functions of the
  adaptive integrators now require the first element of the
  time grid to be equal to the current integrator time
  (`#373 <https://github.com/bluescarni/heyoka/pull/373>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- Move the declarations of all :ref:`keyword arguments <kwargs>`
  into the ``kw.hpp`` header
  (`#372 <https://github.com/bluescarni/heyoka/pull/372>`__).
- The call operators of the event callbacks are not
  ``const`` any more
  (`#369 <https://github.com/bluescarni/heyoka/pull/369>`__).
- **BREAKING**: the minimum supported LLVM version is now 13
  (`#369 <https://github.com/bluescarni/heyoka/pull/369>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- **BREAKING**: heyoka now requires C++20
  (`#369 <https://github.com/bluescarni/heyoka/pull/369>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- **BREAKING**: heyoka now requires fmt>=9
  (`#369 <https://github.com/bluescarni/heyoka/pull/369>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.
- **BREAKING**: heyoka now requires mp++ 1.x
  (`#369 <https://github.com/bluescarni/heyoka/pull/369>`__).
  This is a :ref:`breaking change <bchanges_4_0_0>`.

3.2.0 (2023-11-29)
------------------

New
~~~

- Add step callback set classes to compose step callbacks
  (`#366 <https://github.com/bluescarni/heyoka/pull/366>`__).
- Add support for single-precision computations
  (`#363 <https://github.com/bluescarni/heyoka/pull/363>`__).
- Add model implementing the ELP2000 analytical lunar theory
  (`#362 <https://github.com/bluescarni/heyoka/pull/362>`__).

Changes
~~~~~~~

- When the ``fast_math`` mode is active, the SIMD-vectorised
  mathematical functions now use low-precision implementations.
  This can lead to substantial performance increases in batch mode
  (`#367 <https://github.com/bluescarni/heyoka/pull/367>`__).
- Initialising a step callback or a callable from an empty
  function object (e.g., a null pointer, an empty ``std::function``, etc.)
  now results in an empty object
  (`#366 <https://github.com/bluescarni/heyoka/pull/366>`__).
- Improve performance when creating symbolic expressions for
  large sums and products
  (`#362 <https://github.com/bluescarni/heyoka/pull/362>`__).

3.1.0 (2023-11-13)
------------------

New
~~~

- Implement (leaky) ``ReLU`` and its derivative in the expression
  system (`#357 <https://github.com/bluescarni/heyoka/pull/357>`__,
  `#356 <https://github.com/bluescarni/heyoka/pull/356>`__).
- Add feed-forward neural network model
  (`#355 <https://github.com/bluescarni/heyoka/pull/355>`__).
- Implement the eccentric longitude :math:`F` in the expression
  system (`#352 <https://github.com/bluescarni/heyoka/pull/352>`__).
- Implement the delta eccentric anomaly :math:`\Delta E` in the expression
  system (`#352 <https://github.com/bluescarni/heyoka/pull/352>`__).
  Taylor derivatives are not implemented yet.

Changes
~~~~~~~

- Substantial speedups in the computation of first-order derivatives
  with respect to many variables/parameters
  (`#360 <https://github.com/bluescarni/heyoka/pull/360>`__,
  `#358 <https://github.com/bluescarni/heyoka/pull/358>`__).
- Substantial performance improvements in the computation of
  derivative tensors of large expressions with a high degree
  of internal redundancy
  (`#354 <https://github.com/bluescarni/heyoka/pull/354>`__).

Fix
~~~

- Fix global constants in an LLVM module being generated in unordered fashion
  when compact mode is active. This would result in two logically-identical
  modules being considered different by the in-memory cache
  (`#359 <https://github.com/bluescarni/heyoka/pull/359>`__).
- Fix compiler warning when building without SLEEF support
  (`#356 <https://github.com/bluescarni/heyoka/pull/356>`__).
- Improve the numerical stability of the VSOP2013 model
  (`#353 <https://github.com/bluescarni/heyoka/pull/353>`__).
- Improve the numerical stability of the Kepler solver
  (`#352 <https://github.com/bluescarni/heyoka/pull/352>`__).

3.0.0 (2023-10-07)
------------------

Fix
~~~

- Prevent accidental leaking in the public headers of
  serialisation implementation details
  (`#350 <https://github.com/bluescarni/heyoka/pull/350>`__).
- Fix wrong version compatibility setting in the CMake config-file package
  (`#350 <https://github.com/bluescarni/heyoka/pull/350>`__).
- Work around test failure on ARM + LLVM 17
  (`#350 <https://github.com/bluescarni/heyoka/pull/350>`__).
- Fix orbital elements singularity when using the VSOP2013
  theory at low precision
  (`#348 <https://github.com/bluescarni/heyoka/pull/348>`__).

2.0.0 (2023-09-22)
------------------

New
~~~

- Support LLVM 17 (`#346 <https://github.com/bluescarni/heyoka/pull/346>`__).
- Add model for the circular restricted three-body problem
  (`#345 <https://github.com/bluescarni/heyoka/pull/345>`__).
- heyoka can now automatically vectorise scalar calls to
  floating-point math functions
  (`#342 <https://github.com/bluescarni/heyoka/pull/342>`__).
- The LLVM SLP vectorizer can now be enabled
  (`#341 <https://github.com/bluescarni/heyoka/pull/341>`__).
  This feature is opt-in due to the fact that enabling it
  can considerably increase JIT compilation times.
- Implement an in-memory cache for ``llvm_state``. The cache is used
  to avoid re-optimising and re-compiling LLVM code which has
  already been optimised and compiled during the program execution
  (`#340 <https://github.com/bluescarni/heyoka/pull/340>`__).
- It is now possible to get the LLVM bitcode of
  an ``llvm_state``
  (`#339 <https://github.com/bluescarni/heyoka/pull/339>`__).

Changes
~~~~~~~

- **BREAKING**: the minimum supported LLVM version is now 11
  (`#342 <https://github.com/bluescarni/heyoka/pull/342>`__).
  This is a :ref:`breaking change <bchanges_2_0_0>`.
- The optimisation level for an ``llvm_state`` is now clamped
  within the ``[0, 3]`` range
  (`#340 <https://github.com/bluescarni/heyoka/pull/340>`__).
- The LLVM bitcode is now used internally (instead of the textual
  representation of the IR) when copying and serialising
  an ``llvm_state``
  (`#339 <https://github.com/bluescarni/heyoka/pull/339>`__).
- The optimisation pass in an ``llvm_state`` is now automatically
  called during compilation
  (`#339 <https://github.com/bluescarni/heyoka/pull/339>`__).

Fix
~~~

- Fix compilation in C++20 mode
  (`#340 <https://github.com/bluescarni/heyoka/pull/340>`__).
- Fix the object file of an ``llvm_state`` not being
  preserved during copy and deserialisation
  (`#339 <https://github.com/bluescarni/heyoka/pull/339>`__).
- Fix LLVM module name not being preserved during
  copy and deserialisation of ``llvm_state``
  (`#339 <https://github.com/bluescarni/heyoka/pull/339>`__).
- Fix broken link in the docs.

1.0.0 (2023-08-10)
------------------

New
~~~

- The step callbacks can now optionally implement a ``pre_hook()``
  member function that will be called before the first step
  is taken by a ``propagate_*()`` function
  (`#334 <https://github.com/bluescarni/heyoka/pull/334>`__).
- The heyoka library now passes all ``clang-tidy`` checks
  (`#315 <https://github.com/bluescarni/heyoka/pull/315>`__).
- Introduce several vectorised overloads in the expression
  API. These vectorised overloads allow to perform the same
  operation on a list of expressions more efficiently
  than performing the same operation repeatedly on individual
  expressions
  (`#312 <https://github.com/bluescarni/heyoka/pull/312>`__).
- The expression class is now immutable
  (`#312 <https://github.com/bluescarni/heyoka/pull/312>`__).
- New API to compute high-order derivatives
  (`#309 <https://github.com/bluescarni/heyoka/pull/309>`__).
- The state variables and right-hand side of a system of ODEs
  are now available as read-only properties in the integrator
  classes
  (`#305 <https://github.com/bluescarni/heyoka/pull/305>`__).
- Support LLVM 16.
- New ``model`` module containing ready-made dynamical models
  (`#302 <https://github.com/bluescarni/heyoka/pull/302>`__,
  `#295 <https://github.com/bluescarni/heyoka/pull/295>`__).
- Implement substitution of generic subexpressions
  (`#301 <https://github.com/bluescarni/heyoka/pull/301>`__).
- Add a function to fetch the list of parameters in
  an expression
  (`#301 <https://github.com/bluescarni/heyoka/pull/301>`__).
- The screen output of expressions is now truncated for
  very large expressions
  (`#299 <https://github.com/bluescarni/heyoka/pull/299>`__).

Changes
~~~~~~~

- The step callbacks are now copied in :ref:`ensemble propagations <tut_ensemble>`
  rather than being shared among threads. The aim of this change
  is to reduce the likelihood of data races
  (`#334 <https://github.com/bluescarni/heyoka/pull/334>`__).
- Comprehensive overhaul of the expression system, including:
  enhanced automatic simplification capabilities for sums,
  products and powers, removal of several specialised primitives
  (such as ``square()``, ``neg()``, ``sum_sq()``, etc.),
  re-implementation of division and subtraction as special
  cases of product and sum, and more
  (`#332 <https://github.com/bluescarni/heyoka/pull/332>`__,
  `#331 <https://github.com/bluescarni/heyoka/pull/331>`__,
  `#330 <https://github.com/bluescarni/heyoka/pull/330>`__,
  `#329 <https://github.com/bluescarni/heyoka/pull/329>`__,
  `#328 <https://github.com/bluescarni/heyoka/pull/328>`__,
  `#327 <https://github.com/bluescarni/heyoka/pull/327>`__,
  `#326 <https://github.com/bluescarni/heyoka/pull/326>`__,
  `#325 <https://github.com/bluescarni/heyoka/pull/325>`__,
  `#324 <https://github.com/bluescarni/heyoka/pull/324>`__,
  `#323 <https://github.com/bluescarni/heyoka/pull/323>`__,
  `#322 <https://github.com/bluescarni/heyoka/pull/322>`__).
- Constant folding is now implemented for all functions
  in the expression system
  (`#321 <https://github.com/bluescarni/heyoka/pull/321>`__).
- Moved-from expressions and numbers are now guaranteed to be in the
  default-constructed state
  (`#319 <https://github.com/bluescarni/heyoka/pull/319>`__).
- The expression code has been reorganised into multiple files
  (`#317 <https://github.com/bluescarni/heyoka/pull/317>`__).
- Performance improvements in compact mode for Taylor
  integrators and compiled functions
  (`#303 <https://github.com/bluescarni/heyoka/pull/303>`__).
- Update Catch to version 2.13.10
  (`#301 <https://github.com/bluescarni/heyoka/pull/301>`__).
- The ``get_n_nodes()`` function now returns ``0``
  instead of overflowing
  (`#301 <https://github.com/bluescarni/heyoka/pull/301>`__).
- heyoka now requires Boost >= 1.69
  (`#301 <https://github.com/bluescarni/heyoka/pull/301>`__).
- Performance improvements for several primitives in the
  expression API
  (`#300 <https://github.com/bluescarni/heyoka/pull/300>`__).
- Improve hashing performance for large expressions by
  caching the hashes of repeated subexpressions
  (`#299 <https://github.com/bluescarni/heyoka/pull/299>`__).
- The unstrided version of compiled functions is now forcibly
  inlined, which leads to improved codegen and better performance
  (`#299 <https://github.com/bluescarni/heyoka/pull/299>`__).
- **BREAKING**: the ``make_nbody_sys()`` helper has been replaced by an equivalent
  function in the new ``model`` module
  (`#295 <https://github.com/bluescarni/heyoka/pull/295>`__).
  This is a :ref:`breaking change <bchanges_1_0_0>`.

Fix
~~~

- Work around a likely LLVM bug on ARM
  (`#310 <https://github.com/bluescarni/heyoka/pull/310>`__).
- Fix compilation on OSX when mixing recent libcxx versions with
  old Boost versions
  (`#308 <https://github.com/bluescarni/heyoka/pull/308>`__).
- Do not mix inline member functions with explicit class
  template instantiations. This should fix linking issues
  on Windows when mixing MSVC and clang-cl
  (`#298 <https://github.com/bluescarni/heyoka/pull/298>`__).

0.21.0 (2023-02-16)
-------------------

New
~~~

- Compiled functions now support time-dependent expressions
  (`#294 <https://github.com/bluescarni/heyoka/pull/294>`__).
- The heyoka ABI is now properly versioned and tagged
  (`#290 <https://github.com/bluescarni/heyoka/pull/290>`__).

0.20.1 (2023-01-05)
-------------------

Changes
~~~~~~~

- Mark as visible a couple of internal functions which
  had been marked as hidden by mistake
  (`#286 <https://github.com/bluescarni/heyoka/pull/286>`__).

0.20.0 (2022-12-17)
-------------------

New
~~~

- Add option in the build system to hide the exported LLVM symbols,
  when linking statically
  (`#283 <https://github.com/bluescarni/heyoka/pull/283>`__).
- Add option to force the use of AVX-512 registers
  (`#280 <https://github.com/bluescarni/heyoka/pull/280>`__).
- Implement support for arbitrary-precision computations
  (`#278 <https://github.com/bluescarni/heyoka/pull/278>`__,
  `#276 <https://github.com/bluescarni/heyoka/pull/276>`__).
- Support LLVM 15
  (`#274 <https://github.com/bluescarni/heyoka/pull/274>`__).

Changes
~~~~~~~

- heyoka now depends on CMake >= 3.18 when building from source
  (`#283 <https://github.com/bluescarni/heyoka/pull/283>`__).

Fix
~~~

- Avoid accidental indirect inclusion of libquadmath's header file
  (`#279 <https://github.com/bluescarni/heyoka/pull/279>`__).
- Prevent callbacks from changing the time coordinate of the integrator.
  This was never supported and could lead to crashes and/or hangs
  in the ``propagate_*()`` functions
  (`#278 <https://github.com/bluescarni/heyoka/pull/278>`__).

0.19.0 (2022-09-18)
-------------------

New
~~~

- Add a short tutorial on extended-precision computations
  (`#270 <https://github.com/bluescarni/heyoka/pull/270>`__).
- The numerical integrator classes now support class template argument deduction
  (`#267 <https://github.com/bluescarni/heyoka/pull/267>`__).
- Add the capability to compile multivariate vector functions
  at runtime
  (`#261 <https://github.com/bluescarni/heyoka/pull/261>`__).

Changes
~~~~~~~

- heyoka now builds against recent versions of the fmt library
  without deprecation warnings
  (`#266 <https://github.com/bluescarni/heyoka/pull/266>`__).

Fix
~~~

- Fix compilation against recent LLVM 14.x releases on Windows
  (`#268 <https://github.com/bluescarni/heyoka/pull/268>`__).

0.18.0 (2022-05-11)
-------------------

New
~~~

- Add a timekeeping accuracy benchmark
  (`#254 <https://github.com/bluescarni/heyoka/pull/254>`__).
- Add a function to build (N+1)-body problems
  (`#251 <https://github.com/bluescarni/heyoka/pull/251>`__).
- Support LLVM 14
  (`#247 <https://github.com/bluescarni/heyoka/pull/247>`__).
- Implement :ref:`parallel mode <tut_parallel_mode>`
  for the automatic parallelisation of an individual integration step
  (`#237 <https://github.com/bluescarni/heyoka/pull/237>`__).

Changes
~~~~~~~

- The Kepler solver now returns NaN in case of invalid input arguments
  or if the max number of iterations is exceeded
  (`#252 <https://github.com/bluescarni/heyoka/pull/252>`__).
- heyoka now builds against LLVM 13/14 without deprecation warnings
  (`#242 <https://github.com/bluescarni/heyoka/pull/242>`__).
- In case of an early interruption, the ``propagate_grid()`` function will now
  process all available grid points before the interruption time before exiting
  (`#235 <https://github.com/bluescarni/heyoka/pull/235>`__).
- The ``propagate_grid()`` callbacks are now invoked also if the integration
  is interrupted by a stopping terminal event
  (`#235 <https://github.com/bluescarni/heyoka/pull/235>`__).

Fix
~~~

- Fix several warnings related to variable shadowing when
  compiling in debug mode
  (`#257 <https://github.com/bluescarni/heyoka/pull/257>`__).
- Fix a potential accuracy issue when setting the time coordinate
  in double-length format
  (`#246 <https://github.com/bluescarni/heyoka/pull/246>`__).
- Fix an issue in the ``propagate_grid()`` functions
  that could lead to invalid results in certain corner cases
  (`#234 <https://github.com/bluescarni/heyoka/pull/234>`__).

0.17.1 (2022-02-13)
-------------------

Changes
~~~~~~~

- The ``propagate_for/until()`` callbacks are now invoked also if the integration
  is interrupted by a stopping terminal event
  (`#231 <https://github.com/bluescarni/heyoka/pull/231>`__).

Fix
~~~

- Fix two test failures on FreeBSD
  (`#231 <https://github.com/bluescarni/heyoka/pull/231>`__).

0.17.0 (2022-01-20)
-------------------

New
~~~

- The LLVM version number against which heyoka was built
  is now exported in the CMake config-file package
  (`#225 <https://github.com/bluescarni/heyoka/pull/225>`__).
- It is now possible to access the adaptive integrators'
  time values as double-length floats
  (`#225 <https://github.com/bluescarni/heyoka/pull/225>`__).
- Add support for :ref:`ensemble propagations <tut_ensemble>`
  (`#221 <https://github.com/bluescarni/heyoka/pull/221>`__).
- Several functions in the batch integration API
  now also accept scalar time values in input,
  instead of just vectors. The scalar values
  are automatically splatted into vectors
  of the appropriate size
  (`#221 <https://github.com/bluescarni/heyoka/pull/221>`__).
- Add a function to compute the suggested SIMD size for
  the CPU in use
  (`#220 <https://github.com/bluescarni/heyoka/pull/220>`__).

Changes
~~~~~~~

- Avoid unnecessary copies of the ``propagate_*()`` callbacks
  (`#222 <https://github.com/bluescarni/heyoka/pull/222>`__).

Fix
~~~

- Fix compilation in debug mode when using recent versions
  of ``fmt``
  (`#226 <https://github.com/bluescarni/heyoka/pull/226>`__).
- Fix potential issue arising when certain data structures
  related to event detection are destroyed in the wrong order
  (`#226 <https://github.com/bluescarni/heyoka/pull/226>`__).
- Fix build failures in the benchmark suite
  (`#220 <https://github.com/bluescarni/heyoka/pull/220>`__).

0.16.0 (2021-11-20)
-------------------

New
~~~

- **BREAKING**: add support for :ref:`continuous output <tut_c_output>`
  to the ``propagate_for/until()`` functions
  (`#216 <https://github.com/bluescarni/heyoka/pull/216>`__).
  This is a :ref:`breaking change <bchanges_0_16_0>`.
- Event detection is now available also in batch mode
  (`#214 <https://github.com/bluescarni/heyoka/pull/214>`__).
- Add a sum of squares primitive
  (`#209 <https://github.com/bluescarni/heyoka/pull/209>`__).
- Add new benchmarks and benchmark results to the documentation
  (`#204 <https://github.com/bluescarni/heyoka/pull/204>`__).
- Support LLVM 13
  (`#201 <https://github.com/bluescarni/heyoka/pull/201>`__).

Changes
~~~~~~~

- If ``propagate_grid()`` exits early in batch mode,
  the missing values are now set to NaN instead of zero
  (`#215 <https://github.com/bluescarni/heyoka/pull/215>`__).
- Internal refactoring of the event detection code
  (`#213 <https://github.com/bluescarni/heyoka/pull/213>`__).
- During event detection, improve the performance of the
  fast exclusion check via JIT compilation
  (`#212 <https://github.com/bluescarni/heyoka/pull/212>`__).
- Various internal simplifications in the implementation
  of Taylor derivatives
  (`#208 <https://github.com/bluescarni/heyoka/pull/208>`__).
- Performance optimisations for ODE systems containing large summations
  (`#203 <https://github.com/bluescarni/heyoka/pull/203>`__).
- Performance optimisations in the construction of Taylor integrators
  (`#203 <https://github.com/bluescarni/heyoka/pull/203>`__).
- **BREAKING**: the ``pairwise_sum()`` function has been replaced
  by a new function called ``sum()`` with similar semantics
  (`#203 <https://github.com/bluescarni/heyoka/pull/203>`__).
  This is a :ref:`breaking change <bchanges_0_16_0>`.

Fix
~~~

- Fix various corner-case issues in the integrator classes
  related to data aliasing
  (`#217 <https://github.com/bluescarni/heyoka/pull/217>`__).
- Fix incorrect counting of the number of steps when the
  integration is interrupted by a terminal event
  (`#216 <https://github.com/bluescarni/heyoka/pull/216>`__).

0.15.0 (2021-09-28)
-------------------

New
~~~

- Implement derivatives with respect to the parameters
  (`#196 <https://github.com/bluescarni/heyoka/pull/196>`__).
- Implement additional automatic simplifications in the
  expression system
  (`#195 <https://github.com/bluescarni/heyoka/pull/195>`__).
- Add a way to define symbolic constants in the expression
  system, and implement :math:`\pi` on top of it
  (`#192 <https://github.com/bluescarni/heyoka/pull/192>`__).
- Add a function to compute the size of an expression
  (`#189 <https://github.com/bluescarni/heyoka/pull/189>`__).
- Quadruple precision is now correctly supported on PPC64
  (`#188 <https://github.com/bluescarni/heyoka/pull/188>`__).
- Add an implementation of the VSOP2013 analytical solution
  for the motion of the planets of the Solar System, usable
  in the definition of differential equations
  (`#186 <https://github.com/bluescarni/heyoka/pull/186>`__,
  `#183 <https://github.com/bluescarni/heyoka/pull/183>`__,
  `#180 <https://github.com/bluescarni/heyoka/pull/180>`__).
- Add the two-argument inverse tangent function ``atan2()``
  to the expression system
  (`#182 <https://github.com/bluescarni/heyoka/pull/182>`__).
- Implement additional automatic simplifications for sin/cos
  (`#180 <https://github.com/bluescarni/heyoka/pull/180>`__).

Changes
~~~~~~~

- Implement a fast exclusion check for event detection which
  improves performance when no event triggers in a timestep
  (`#198 <https://github.com/bluescarni/heyoka/pull/198>`__).
- **BREAKING**: the function class now uses reference
  semantics. This means that copy operations on
  non-trivial expressions now result in shallow copies,
  not deep copies. This is a :ref:`breaking change <bchanges_0_15_0>`
  (`#192 <https://github.com/bluescarni/heyoka/pull/192>`__).
- heyoka now depends on the `TBB <https://github.com/oneapi-src/oneTBB>`__ library
  (`#186 <https://github.com/bluescarni/heyoka/pull/186>`__).

Fix
~~~

- Don't force the use of static MSVC runtime when
  compiling heyoka as a static library
  (`#198 <https://github.com/bluescarni/heyoka/pull/198>`__).
- Fix compilation as a static library
  (`#195 <https://github.com/bluescarni/heyoka/pull/195>`__).
- Various fixes to the PPC64 support
  (`#188 <https://github.com/bluescarni/heyoka/pull/188>`__,
  `#187 <https://github.com/bluescarni/heyoka/pull/187>`__).
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
