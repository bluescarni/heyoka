Known issues
============

Unsolved
========

* Under very specific circumstances, C++ code executed right after
  code that was JIT-compiled by heyoka might produce nonsensical results.
  This happens only if **all** the following conditions are met:

  * you are on an Intel x86 platform where ``long double`` corresponds
    to the extended-precision 80-bit x86 floating-point type,
  * heyoka was compiled with support for quadruple-precision computations
    via :cpp:class:`mppp::real128`,
  * JIT-compiled code using **both** 80-bit and quadruple-precision datatypes
    was executed,
  * the "fast math" flag was enabled during JIT compilation.

  The root cause is most likely a code-generation issue in LLVM. This issue
  is currently under investigation.

Solved
======

* In several LLVM versions, attempting to use :ref:`batch mode <tut_batch_mode>`
  with the extended precision ``long double`` type on x86 processors will lead
  to incorrect results. This is due to code generation issues in LLVM with
  ``long double`` vector types. This problem seems to have been rectified in
  LLVM 18. Note that, in practice, there is no reason to attempt to use batch
  mode with ``long double`` as currently there are no CPUs implementing SIMD operations
  on extended-precision datatypes.
* Due to an upstream bug, if you compile heyoka linking statically against LLVM 17
  while enabling the ``HEYOKA_HIDE_LLVM_SYMBOLS`` option (see the
  :ref:`installation instructions <installation_from_source>`), you may experience
  runtime errors due to missing symbols. This problem should be fixed in LLVM 18.
  A patch fixing the issue in LLVM 17
  is available `here <https://github.com/llvm/llvm-project/commit/122ebe3b500190b1f408e2e6db753853e297ba28>`__.
* Due to an `upstream bug <https://github.com/conda-forge/mpfr-feedstock/issues/44>`__,
  multiprecision :ref:`ensemble propagations <tut_ensemble>`
  crash on OSX arm64 when using heyoka's conda-forge package. This is due to the conda-forge
  package for the MPFR library not being compiled in thread-safe mode. The solution is to update
  to the latest version of the MPFR package, which includes a fix for this issue.
