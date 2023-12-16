.. _tut_s11n:

Serialisation
=============

.. versionadded:: 0.12.0

Starting with version 0.12.0, heyoka supports serialisation via the
`Boost.Serialization <https://www.boost.org/doc/libs/release/libs/serialization/doc/index.html>`__
library. Before showing a couple of examples of serialisation in action,
we need to emphasise a couple of very important **caveats**:

* currently, heyoka supports serialisation only to/from binary archives;
* the serialisation format is platform-dependent and it also depends
  on the versions of heyoka, LLVM and Boost. Thus, the serialised
  representation of heyoka objects is **not portable** across platforms
  or across different versions of heyoka or its dependencies. Do **not**
  try to use the serialised representation of heyoka objects as an exchange
  format, as this will result in undefined behaviour;
* heyoka does not make any attempt to validate the state of a deserialised object.
  Thus, a maliciously-crafted binary archive could be used
  to crash heyoka or even execute arbitrary code on the machine.

The last point is particularly important: because the integrator objects
contain blobs of binary code,
a maliciously-crafted archive can easily be used
to execute arbitrary code on the host machine.

Let us repeat again these warnings for visibility:

.. warning::

   Do **not** load heyoka objects from untrusted archives, as this could lead
   to the execution of malicious code.

   Do **not** use heyoka archives as a data exchange format, and make sure that
   all the archives you load from have been produced with the same versions of heyoka,
   LLVM and Boost that you are currently using.

With these warnings out of the way, let us proceed to the code.

A simple example
----------------

In order to illustrate the (de)serialisation workflow, we will be using
our good old friend, the simple pendulum. We begin as usual with the definition
of the symbolic variables and the integrator object:

.. literalinclude:: ../tutorial/s11n_basic.cpp
    :language: c++
    :lines: 18-29

We then integrate for a few timesteps, so that the time coordinate and the state
will evolve from their initial values:

.. literalinclude:: ../tutorial/s11n_basic.cpp
    :language: c++
    :lines: 31-39

.. code-block:: console

   ta time (original)     : 1.04348
   ta state (original)    : [-0.0506049, -0.00537327]

Let us then proceed to the serialisation of the integrator object into
a binary archive. For the purpose of this tutorial we will be writing the archive
into a string stream, but the same code would work for serialisation into a file
object or into any other standard C++ stream:

.. literalinclude:: ../tutorial/s11n_basic.cpp
    :language: c++
    :lines: 41-46

Note how we bracket the lifetime of the binary archive ``oa`` by placing it into a
separate scope: the invocation of the destructor of ``oa`` at the end of the block
will ensure that the data is written to the stream ``ss``. Please refer to the
`Boost.Serialization docs <https://www.boost.org/doc/libs/release/libs/serialization/doc/index.html>`__
for more information about the serialisation API.

After having serialised ``ta`` into ``ss``, we reset ``ta`` to its initial state, and we
print the time and the state vector to screen in order to confirm that ``ta`` has indeed
been reset:

.. literalinclude:: ../tutorial/s11n_basic.cpp
    :language: c++
    :lines: 48-52

.. code-block:: console

   ta time (after reset)  : 0
   ta state (after reset) : [0.05, 0.025]

We are now ready to recover the serialised representation of ``ta`` from the string stream. After
loading ``ta`` from the archive, we will print the time and the state vector to screen to confirm that,
indeed, the previous state of ``ta`` has been correctly recovered:

.. literalinclude:: ../tutorial/s11n_basic.cpp
    :language: c++
    :lines: 54-63

.. code-block:: console

   ta time (from archive) : 1.04348
   ta state (from archive): [-0.0506049, -0.00537327]

Full code listing
^^^^^^^^^^^^^^^^^
   
.. literalinclude:: ../tutorial/s11n_basic.cpp
    :language: c++
    :lines: 9-

Serialising event callbacks
---------------------------

The serialisation of integrator objects in the presence
of :ref:`events <tut_events>` needs special attention, because
the event callbacks are internally implemented as type-erased classes
on top of a traditional object-oriented hierarchy. In other words, an integrator
stores the callbacks as pointers to a base class, and in this situation
the Boost.Serialization library needs some extra assistance in order to work
correctly.

There are two steps that need to be taken in order to enable the
serialisation of event callbacks:

* make the callback itself serialisable. The `Boost.Serialization docs <https://www.boost.org/doc/libs/release/libs/serialization/doc/index.html>`__
  explain in detail how to add serialisation capabilities to a class.
  Note that, in order to be serialisable, a callback must be implemented
  as a function object - it is not possible to serialise function pointers
  or lambdas;
* register the callback in the serialisation system via the invocation
  of a macro (see below).

Let us see a concrete example. We begin with the definition of a simple callback
class:

.. literalinclude:: ../tutorial/s11n_event.cpp
    :language: c++
    :lines: 16-28

This trivial callback function object is meant to be used in a non-terminal event.
In order to make the callback serialisable we add
a member function template called ``serialize()`` which, in this specific case,
also does not perform any action because the callback has no state. If the callback
contained data members, we would need to serialise them one by one - see the
`Boost.Serialization docs <https://www.boost.org/doc/libs/release/libs/serialization/doc/index.html>`__
for details about adding serialisation capabilities to a class.

After having added serialisation capabilities to our ``callback``, we need to register
it in heyoka's serialisation system. This is accomplished through the use of the
``HEYOKA_S11N_CALLABLE_EXPORT()`` macro:

.. literalinclude:: ../tutorial/s11n_event.cpp
    :language: c++
    :lines: 30-31

The ``HEYOKA_S11N_CALLABLE_EXPORT()`` macro takes as first input argument the name of the class
being registered (``callback`` in this case). The remaining arguments are the signature
of the callback: ``void`` is the return type, ``taylor_adaptive<double> &``, ``double``
and ``int`` its argument types. Note that this macro must be invoked in the
root namespace and all arguments should be spelled out as fully-qualified
names (in this example we can avoid the extra typing due to the ``using namespace heyoka``
statement).

The ``callback`` class is now ready to be (de)serialised. Let us see a simple
example, again based on the simple pendulum:

.. literalinclude:: ../tutorial/s11n_event.cpp
    :language: c++
    :lines: 35-69

.. code-block:: console

   Number of events (original)    : 1
   Number of events (from archive): 1

The screen output indeed confirms that the event callback was correctly (de)serialised.
If we had not used the ``HEYOKA_S11N_CALLABLE_EXPORT()`` macro to register the callback, a runtime
exception would have been raised during the serialisation of the integrator object.

Full code listing
^^^^^^^^^^^^^^^^^
   
.. literalinclude:: ../tutorial/s11n_event.cpp
    :language: c++
    :lines: 9-
