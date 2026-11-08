Known issues
============

* Under very specific circumstances, C++ code executed right after
  code that was JIT-compiled by heyoka might produce nonsensical results.
  This happens only if **all** the following conditions are met:

  * you are on an Intel x86 platform where ``long double`` corresponds
    to the extended-precision 80-bit x86 floating-point type,
  * heyoka was compiled with support for quadruple-precision computations
    via :cpp:class:`mppp::real128`,
  * JIT-compiled code using **both** 80-bit and quadruple-precision datatypes
    was executed,
  * the ``fast_math`` flag was enabled during JIT compilation.

  The root cause is most likely a code-generation/optimisation problem in LLVM.
  This issue is currently under investigation.
* The option for selecting the code used model for JIT compilation
  (added in heyoka 6.0.0) is currently disabled on Windows due to what
  looks like an LLVM bug. The issue is currently under investigation.
