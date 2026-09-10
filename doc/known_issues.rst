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
* The parallel compilation feature (initially added in heyoka 6.0.0) is currently disabled.
  The reason is a likely thread scheduling bug in LLVM's parallel compilation facilities
  which, on Unix systems, rarely results in a multiply-defined symbol, ultimately leading to a compilation
  failure. The issue is currently under investigation by the LLVM developers.
* The option for selecting the code used model for JIT compilation
  (added in heyoka 6.0.0) is currently disabled on Windows due to what
  looks like an LLVM bug. The issue is currently under investigation.
