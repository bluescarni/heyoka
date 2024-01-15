Known issues
============

* Due to an upstream bug, if you compile heyoka linking statically against LLVM 17
  while enabling the ``HEYOKA_HIDE_LLVM_SYMBOLS`` option (see the
  :ref:`installation instructions <installation_from_source>`), you may experience
  runtime errors due to missing symbols. This problem should be fixed in LLVM 18.
  A patch fixing the issue in LLVM 17
  is available `here <https://github.com/llvm/llvm-project/commit/122ebe3b500190b1f408e2e6db753853e297ba28>`__.
