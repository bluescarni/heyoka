Changelog
=========

0.3.0 (unreleased)
------------------

New
~~~

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
  (`#76 <https://github.com/bluescarni/heyoka/pull/76>`__,
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
