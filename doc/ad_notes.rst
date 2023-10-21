.. _ad_notes:

Notes on automatic differentiation
==================================

Preliminaries
-------------

Definition of normalised derivative:

.. math::
   :label: eq_norm_der_00

   x^{\left[ n\right]}\left( t \right) = \frac{1}{n!} x^{\left( n\right)}\left( t \right).

`General Leibniz rule <https://en.wikipedia.org/wiki/General_Leibniz_rule>`__: given
:math:`a\left( t \right) = b\left( t \right) c\left( t \right)`, then

.. math::
   :label: eq_leibniz_00

   a^{\left[ n\right]}\left( t \right) = \sum_{j=0}^n b^{\left[ n - j\right]}\left( t \right) c^{\left[ j\right]}\left( t \right).

Basic arithmetic
----------------

Addition and subtraction
^^^^^^^^^^^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = b\left( t \right) \pm c\left( t \right)`, trivially

.. math::
   :label: eq_ad_addsub_00

   a^{\left[ n \right]}\left( t \right) = b^{\left[ n \right]}\left( t \right) \pm c^{\left[ n \right]}\left( t \right).

Multiplication
^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = b\left( t \right) c\left( t \right)`, the derivative :math:`a^{\left[ n \right]}\left( t \right)`
is given directly by the application of the general Leibniz rule :eq:`eq_leibniz_00`.

Division
^^^^^^^^

Given :math:`a\left( t \right) = \frac{b\left( t \right)}{c\left( t \right)}`, we can write

.. math::
   :label:

   a\left( t \right) c\left( t \right) = b\left( t \right).

We can now apply the normalised derivative of order :math:`n` to both sides, use :eq:`eq_leibniz_00` and re-arrange to obtain:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{c^{\left[ 0 \right]}\left( t \right)}\left[ b^{\left[ n \right]}\left( t \right) - \sum_{j=1}^n a^{\left[ n - j \right]}\left( t \right) c^{\left[ j \right]}\left( t \right)\right].

Squaring
--------

Given :math:`a\left( t \right) = b\left( t \right)^2`, the computation of :math:`a^{\left[ n \right]}\left( t \right)` is a special
case of :eq:`eq_leibniz_00` in which we take advantage of the summation's symmetry in order to halve the computational
complexity:

.. math::
   :label: eq_ad_square_00

   a^{\left[ n \right]}\left( t \right) =
   \begin{cases}
   2\sum_{j=0}^{\frac{n}{2}-1} b^{\left[ n - j \right]}\left( t \right) b^{\left[ j \right]}\left( t \right) + \left( b^{\left[ \frac{n}{2} \right]}\left( t \right) \right)^2 \mbox{ if $n$ is even}, \\
   2\sum_{j=0}^{\frac{n-1}{2}} b^{\left[ n - j \right]}\left( t \right) b^{\left[ j \right]}\left( t \right) \mbox{ if $n$ is odd}.
   \end{cases}

Square root
-----------

Given :math:`a\left( t \right) =\sqrt{b\left( t \right)}`, we can write

.. math::
   :label:

   a\left( t \right)^2 = b\left( t \right).

We can apply the normalised derivative of order :math:`n` to both sides, and, with the help of :eq:`eq_ad_square_00`, we obtain:

.. math::
   :label:

   b^{\left[ n \right]}\left( t \right) =
   \begin{cases}
   2\sum_{j=0}^{\frac{n}{2}-1} a^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right) + \left( a^{\left[ \frac{n}{2} \right]}\left( t \right) \right)^2 \mbox{ if $n$ is even}, \\
   2\sum_{j=0}^{\frac{n-1}{2}} a^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right) \mbox{ if $n$ is odd}.
   \end{cases}

We can then isolate :math:`a^{\left[ n  \right]}\left( t \right)` to obtain:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) =
   \begin{cases}
   \frac{1}{2a^{\left[ 0 \right]}\left( t \right)} \left[ b^{\left[ n \right]}\left( t \right) - 2\sum_{j=1}^{\frac{n}{2}-1} a^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right) - \left( a^{\left[ \frac{n}{2} \right]}\left( t \right) \right)^2 \right] \mbox{ if $n$ is even}, \\
   \frac{1}{2a^{\left[ 0 \right]}\left( t \right)} \left[ b^{\left[ n \right]}\left( t \right) - 2\sum_{j=0}^{\frac{n-1}{2}} a^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right) \right] \mbox{ if $n$ is odd}.
   \end{cases}

Exponentiation
--------------

Given :math:`a\left( t \right) = b\left( t \right)^\alpha`, with :math:`\alpha \neq 0`, we have

.. math::
   :label:

   a^\prime\left( t \right) = \alpha b\left( t \right)^{\alpha - 1} b^\prime\left( t \right).

By multiplying both sides by :math:`b\left( t \right)` we obtain

.. math::
   :label:

   \begin{aligned}
   b\left( t \right) a^\prime\left( t \right) & = b\left( t \right) \alpha b\left( t \right)^{\alpha - 1} b^\prime\left( t \right) \\
   & = \alpha  b^\prime\left( t \right) a\left( t \right).
   \end{aligned}

We can now apply the normalised derivative of order :math:`n-1` to both sides, use :eq:`eq_leibniz_00` and re-arrange to obtain,
for :math:`n > 0`:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{n b^{\left[ 0 \right]}\left( t \right)} \sum_{j=0}^{n-1} \left[ n\alpha - j \left( \alpha + 1 \right) \right] b^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right).

Exponentials
------------

Natural exponential
^^^^^^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = e^{b\left( t \right)}`, we have

.. math::
   :label:

   a^\prime\left( t \right) = e^{b\left( t \right)}b^\prime\left( t \right) = a\left( t \right) b^\prime\left( t \right).

We can now apply the normalised derivative of order :math:`n-1` to both sides, use :eq:`eq_norm_der_00` and :eq:`eq_leibniz_00`
and obtain, for :math:`n > 0`:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{n} \sum_{j=1}^{n} j a^{\left[ n - j \right]}\left( t \right) b^{\left[ j \right]}\left( t \right).

Standard logistic function
^^^^^^^^^^^^^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = \operatorname{sig} {b\left( t \right)}`, where :math:`\operatorname{sig}\left( x \right)`
is the `standard logistic function <https://en.wikipedia.org/wiki/Logistic_function>`__

.. math::
   :label:

   \operatorname{sig} \left( x \right) = \frac{1}{1+e^{-x}},

we have

.. math::
   :label:

   a^\prime\left( t \right) = \operatorname{sig}{b\left( t \right)} \left[1 - \operatorname{sig}{b\left( t \right)} \right] b^\prime\left( t \right) = a\left( t \right) \left[1 - a\left( t \right) \right] b^\prime\left( t \right),

which, after the introduction of the auxiliary function

.. math::
   :label:

   c\left( t \right)  = a^2\left( t \right) ,

becomes

.. math::
   :label:

   a^\prime\left( t \right) = \left[ a\left( t \right) - c\left( t \right) \right] b^\prime\left( t \right).

After applying the normalised derivative of order :math:`n-1` to both sides, we can use :eq:`eq_norm_der_00`,
:eq:`eq_leibniz_00` and :eq:`eq_ad_addsub_00` to obtain, for :math:`n > 0`:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{n}\sum_{j=1}^{n} j \left[ a^{\left[ n - j \right]} \left( t \right)- c^{\left[ n - j \right]}\left( t \right)\right] b^{\left[ j \right]}\left( t \right).


Logarithms
----------

Natural logarithm
^^^^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = \log b\left( t \right)`, we have

.. math::
   :label:

   a^\prime\left( t \right) = \frac{b^\prime\left( t \right)}{b\left( t \right)},

or, equivalently,

.. math::
   :label:

   b\left( t \right) a^\prime\left( t \right) = b^\prime\left( t \right).

We can now apply the normalised derivative of order :math:`n-1` to both sides, use :eq:`eq_norm_der_00` and :eq:`eq_leibniz_00`
and re-arrange to obtain, for :math:`n > 0`:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{n b^{\left[ 0 \right]}\left( t \right)} \left[ n b^{\left[ n \right]}\left( t \right) - \sum_{j=1}^{n-1} j b^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right) \right].

Trigonometric functions
-----------------------

.. _ad_tan:

Tangent
^^^^^^^

Given :math:`a\left( t \right) = \tan b\left( t \right)`, we have

.. math::
   :label:

   a^\prime\left( t \right) = \left[ \tan^2 b\left( t \right) + 1 \right] b^\prime\left( t \right) = a^2\left( t \right)b^\prime\left( t \right) + b^\prime\left( t \right),

which, after the introduction of the auxiliary function

.. math::
   :label:

   c\left( t \right)  = a^2\left( t \right) ,

becomes

.. math::
   :label:

   a^\prime\left( t \right) = c\left( t \right) b^\prime\left( t \right) + b^\prime\left( t \right).

After applying the normalised derivative of order :math:`n-1` to both sides, we can use :eq:`eq_norm_der_00`,
:eq:`eq_leibniz_00` and :eq:`eq_ad_addsub_00` to obtain, for :math:`n > 0`:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{n}\sum_{j=1}^{n} j c^{\left[ n - j \right]}\left( t \right) b^{\left[ j \right]}\left( t \right) + b^{\left[ n \right]}\left( t \right).

Inverse trigonometric functions
-------------------------------

.. _ad_asin:

Inverse sine
^^^^^^^^^^^^

Given :math:`a\left( t \right) = \arcsin b\left( t \right)`, we have

.. math::
   :label:

   a^\prime\left( t \right) = \frac{b^\prime\left( t \right)}{\sqrt{1 - b^2\left( t \right) }},

or, equivalently,

.. math::
   :label: eq_ad_asin00

   a^\prime\left( t \right) \sqrt{1 - b^2\left( t \right) } = b^\prime\left( t \right).

We introduce the auxiliary function

.. math::
   :label:

   c\left( t \right)  = \sqrt{1 - b^2\left( t \right) },

so that :eq:`eq_ad_asin00` can be rewritten as

.. math::
   :label:

   a^\prime\left( t \right) c\left( t \right)  = b^\prime\left( t \right).

Applying the normalised derivative of order :math:`n-1` to both sides yields, via :eq:`eq_norm_der_00`:

.. math::
   :label:

   \left[a^\prime\left( t \right) c\left( t \right)\right]^{\left[ n - 1 \right]}  = n b^{\left[ n \right]} \left( t \right).

We can now apply the general Leibniz rule :eq:`eq_leibniz_00` to the left-hand side and re-arrange
the terms to obtain, for :math:`n > 0`:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{n c^{\left[ 0 \right]}\left( t \right)}\left[ n b^{\left[ n \right]}\left( t \right) - \sum_{j=1}^{n-1} j c^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right) \right].

.. _ad_acos:

Inverse cosine
^^^^^^^^^^^^^^

The derivation is identical to the :ref:`inverse sine <ad_asin>`, apart from a sign change.
Given :math:`a\left( t \right) = \arccos b\left( t \right)`,
the final result is, for :math:`n > 0`:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = -\frac{1}{n c^{\left[ 0 \right]}\left( t \right)}\left[ n b^{\left[ n \right]}\left( t \right) + \sum_{j=1}^{n-1} j c^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right) \right],

with :math:`c\left( t \right)` defined as:

.. math::
   :label:

   c\left( t \right)  = \sqrt{1 - b^2\left( t \right) }.

.. _ad_atan:

Inverse tangent
^^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = \arctan b\left( t \right)`, we have

.. math::
   :label:

   a^\prime\left( t \right) = \frac{b^\prime\left( t \right)}{1 + b^2\left( t \right) },

or, equivalently,

.. math::
   :label: eq_ad_atan00

   a^\prime\left( t \right) \left[1 + b^2\left( t \right) \right] = b^\prime\left( t \right).

We introduce the auxiliary function

.. math::
   :label:

   c\left( t \right)  = b^2\left( t \right),

so that :eq:`eq_ad_atan00` can be rewritten as

.. math::
   :label:

   a^\prime\left( t \right) + a^\prime\left( t \right) c\left( t \right)  = b^\prime\left( t \right).

Applying the normalised derivative of order :math:`n-1` to both sides yields, via :eq:`eq_norm_der_00` and :eq:`eq_ad_addsub_00`:

.. math::
   :label:

   n a^{\left[ n \right]} \left( t \right) + \left[a^\prime\left( t \right) c\left( t \right)\right]^{\left[ n - 1 \right]}  = n b^{\left[ n \right]} \left( t \right).

With the help of the general Leibniz rule :eq:`eq_leibniz_00`, after re-arranging we obtain, for :math:`n > 0`:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{n \left[ c^{\left[ 0 \right]}\left( t \right) + 1 \right]}\left[ n b^{\left[ n \right]}\left( t \right) - \sum_{j=1}^{n-1} j c^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right) \right].

Two-argument inverse tangent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = \operatorname{arctan2}\left( b\left( t \right), c\left( t \right) \right)`, we have

.. math::
   :label: eq_ad_atan200

   a^\prime\left( t \right) = \frac{c\left( t \right) b^\prime\left( t \right)-b\left( t \right)c^\prime \left( t \right)}
   {b^2\left( t \right)+c^2\left( t \right)}.

After the introduction of the auxiliary function

.. math::
   :label:

   d\left( t \right)  = b^2\left( t \right)+c^2\left( t \right),

:eq:`eq_ad_atan200` can be rewritten as

.. math::
   :label:

   d\left( t \right)a^\prime\left( t \right) = c\left( t \right) b^\prime\left( t \right)-b\left( t \right)c^\prime \left( t \right).

We can now apply the normalised derivative of order :math:`n-1` to both sides, and, via :eq:`eq_leibniz_00`, obtain, for :math:`n > 0`:

.. math::
   :label:

   \begin{aligned}
   a^{\left[ n \right]}\left( t \right) &= \frac{1}{nd^{\left[ 0 \right]}\left( t \right)}\left[\vphantom{\sum_{j=1}^{n-1}j\left( \right)}
   n\left( c^{\left[ 0 \right]}\left( t \right) b^{\left[ n \right]}\left( t \right) - b^{\left[ 0 \right]}\left( t \right) c^{\left[ n \right]}\left( t \right)\right) \right.\\
   &\left. + \sum_{j=1}^{n-1}j\left( c^{\left[ n-j \right]}\left( t \right) b^{\left[ j \right]}\left( t \right) -
   b^{\left[ n-j \right]}\left( t \right) c^{\left[ j \right]}\left( t \right) -
   d^{\left[ n-j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right) \right) \right].
   \end{aligned}

Hyperbolic functions
--------------------

.. _ad_sinh:

Hyperbolic sine
^^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = \sinh b\left( t \right)`, we have

.. math::
   :label: eq_ad_sinh_00

   a^\prime\left( t \right) = b^\prime\left( t \right) \cosh b\left( t \right).

We introduce the auxiliary function

.. math::
   :label:

   c\left( t \right) = \cosh b\left( t \right),

so that :eq:`eq_ad_sinh_00` can be rewritten as

.. math::
   :label:

   a^\prime\left( t \right) = c\left( t \right) b^\prime\left( t \right).

We can now apply the normalised derivative of order :math:`n-1` to both sides, and, via :eq:`eq_leibniz_00`, obtain, for :math:`n > 0`:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right)  = \frac{1}{n} \sum_{j=1}^{n} j c^{\left[ n - j \right]}\left( t \right) b^{\left[ j \right]}\left( t \right).

Hyperbolic cosine
^^^^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = \cosh b\left( t \right)`, the process of deriving :math:`a^{\left[ n \right]}\left( t \right)` is
identical to the :ref:`hyperbolic sine <ad_sinh>`. After the definition of the auxiliary function

.. math::
   :label:

   s\left( t \right) = \sinh b\left( t \right),

the final result, for :math:`n > 0`, is:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right)  = \frac{1}{n} \sum_{j=1}^{n} j s^{\left[ n - j \right]}\left( t \right) b^{\left[ j \right]}\left( t \right).

Hyperbolic tangent
^^^^^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = \tanh b\left( t \right)`, the process of deriving :math:`a^{\left[ n \right]}\left( t \right)` is
identical to the :ref:`tangent <ad_tan>`, apart from a sign change. After the definition of the auxiliary function

.. math::
   :label:

   c\left( t \right)  = a^2\left( t \right),

the final result, for :math:`n > 0`, is:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = b^{\left[ n \right]}\left( t \right) - \frac{1}{n}\sum_{j=1}^{n} j c^{\left[ n - j \right]}\left( t \right) b^{\left[ j \right]}\left( t \right).

Inverse hyperbolic functions
----------------------------

.. _ad_asinh:

Inverse hyperbolic sine
^^^^^^^^^^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = \operatorname{arsinh} b\left( t \right)`, the process of deriving :math:`a^{\left[ n \right]}\left( t \right)` is
identical to the :ref:`inverse sine <ad_asin>`, apart from a sign change. After the definition of the auxiliary function

.. math::
   :label:

   c\left( t \right)  = \sqrt{1 + b^2\left( t \right) },

the final result, for :math:`n > 0`, is:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{n c^{\left[ 0 \right]}\left( t \right)}\left[ n b^{\left[ n \right]}\left( t \right) - \sum_{j=1}^{n-1} j c^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right) \right].

Inverse hyperbolic cosine
^^^^^^^^^^^^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = \operatorname{arcosh} b\left( t \right)`, the process of deriving :math:`a^{\left[ n \right]}\left( t \right)` is
identical to the :ref:`inverse hyperbolic sine <ad_asinh>`, apart from a sign change. After the definition of the auxiliary function

.. math::
   :label:

   c\left( t \right)  = \sqrt{b^2\left( t \right) - 1 },

the final result, for :math:`n > 0`, is:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{n c^{\left[ 0 \right]}\left( t \right)}\left[ n b^{\left[ n \right]}\left( t \right) - \sum_{j=1}^{n-1} j c^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right) \right].

Inverse hyperbolic tangent
^^^^^^^^^^^^^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = \operatorname{artanh} b\left( t \right)`, the process of deriving :math:`a^{\left[ n \right]}\left( t \right)` is
identical to the :ref:`inverse tangent <ad_atan>`, apart from a sign change. After the definition of the auxiliary function

.. math::
   :label:

   c\left( t \right)  = b^2\left( t \right),

the final result, for :math:`n > 0`, is:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{n \left[1 - c^{\left[ 0 \right]}\left( t \right) \right]}\left[ n b^{\left[ n \right]}\left( t \right) + \sum_{j=1}^{n-1} j c^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right) \right].

Special functions
-----------------

.. _ad_erf:

Error function
^^^^^^^^^^^^^^

Given :math:`a\left( t \right) = \operatorname{erf} b\left( t \right)`, we have

.. math::
   :label:

   a^\prime\left( t \right) = \frac 2{\sqrt\pi} \exp{\left[-b^2\left( t \right)\right]} b^\prime\left( t \right),

which, after the introduction of the auxiliary function

.. math::
   :label:

   c\left( t \right)  = \exp{\left[ -b^2\left( t \right)\right]} ,

becomes

.. math::
   :label:

   a^\prime\left( t \right) = \frac 2{\sqrt\pi}c\left( t \right) b^\prime\left( t \right).

After applying the normalised derivative of order :math:`n-1` to both sides, we can use :eq:`eq_norm_der_00`
and :eq:`eq_leibniz_00` to obtain, for :math:`n > 0`:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac 1n \frac 2{\sqrt\pi}\sum_{j=1}^{n} j c^{\left[ n - j \right]}\left( t \right) b^{\left[ j \right]}\left( t \right).

Celestial mechanics
-------------------

.. _kepE_ad:

Kepler's eccentric anomaly
^^^^^^^^^^^^^^^^^^^^^^^^^^

The `eccentric anomaly <https://en.wikipedia.org/wiki/Eccentric_anomaly>`__ is the bivariate function :math:`E = E\left( e, M \right)` implicitly defined by the
trascendental equation

.. math::
   :label:

   M = E - e \sin E,

with :math:`e \in \left[ 0, 1 \right)`. Given :math:`a\left( t \right) = E\left( e\left( t \right), M \left( t \right) \right)`, we have

.. math::
   :label:

   a^\prime\left( t \right) = \frac{\partial E}{\partial e}e^\prime\left( t \right) + \frac{\partial E}{\partial M}M^\prime\left( t \right),

where the partial derivatives are

.. math::
   :label:

   \begin{cases}
   \frac{\partial E}{\partial e} = \frac{\sin E}{1-e\cos E}, \\
   \frac{\partial E}{\partial M} = \frac{1}{1-e\cos E}. \\
   \end{cases}

Expanding the partial derivatives yields

.. math::
   :label:

   a^\prime\left( t \right) = \frac{e^\prime \left( t \right)\sin a\left(t \right) + M^\prime\left( t \right)}{1-e\left( t \right)\cos a\left(t \right)},

or, equivalently,

.. math::
   :label: eq_ad_kepE_00

   a^\prime\left( t \right) -  a^\prime\left( t \right) e \left( t \right) \cos a\left(t \right) = e^\prime \left( t \right)\sin a\left(t \right) + M^\prime\left( t \right).

We can now introduce the auxiliary functions

.. math::
   :label:

   \begin{cases}
   c\left( t \right) = e\left( t \right) \cos a\left(t \right), \\
   d\left( t \right) = \sin a\left(t \right), \\
   \end{cases}

so that :eq:`eq_ad_kepE_00` can be rewritten as

.. math::
   :label:

   a^\prime\left( t \right) -  a^\prime\left( t \right) c\left( t \right) = e^\prime \left( t \right)d\left(t \right) + M^\prime\left( t \right).

After applying the normalised derivative of order :math:`n-1` to both sides, we can use :eq:`eq_norm_der_00`
and :eq:`eq_leibniz_00` and re-arrange to obtain, for :math:`n > 0`:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{n \left(1 - c^{\left[ 0 \right]}\left( t \right)\right)}
   \left[
   n\left( e^{\left[ n \right]}\left( t \right) d^{\left[ 0 \right]}\left( t \right) + M^{\left[ n \right]}\left( t \right)\right) +
   \sum_{j=1}^{n-1}j\left( c^{\left[ n - j \right]}\left( t \right) a^{\left[ j \right]}\left( t \right)+
   d^{\left[ n - j \right]}\left( t \right)e^{\left[ j \right]}\left( t \right)
   \right)
   \right].

Eccentric longitude
^^^^^^^^^^^^^^^^^^^

The `eccentric longitude <https://articles.adsabs.harvard.edu//full/1972CeMec...5..303B/0000309.000.html>`__
is the trivariate function :math:`F = F\left( h, k, \lambda \right)` implicitly defined by the
trascendental equation

.. math::
   :label:

   \lambda = F + h\cos F - k\sin F,

with :math:`h^2+k^2 < 1`. Given :math:`a\left( t \right) = F\left( h\left( t \right), k \left( t \right), \lambda \left( t \right) \right)`,
we have

.. math::
   :label:

   a^\prime\left( t \right) = \frac{k^\prime \left( t \right)\sin a\left(t \right) - h^\prime \left( t \right)\cos a\left(t \right)
   + \lambda^\prime\left( t \right)}{1-h\left( t \right)\sin a\left(t \right) -k\left( t \right)\cos a\left(t \right)}.

After the introduction of the auxiliary functions

.. math::
   :label:

   \begin{cases}
   c\left( t \right) = h\left( t \right) \sin a\left(t \right), \\
   d\left( t \right) = k\left( t \right) \cos a\left(t \right), \\
   e\left( t \right) = \sin a\left(t \right), \\
   f\left( t \right) = \cos a\left(t \right), \\
   \end{cases}

we can then proceed in the same way as explained for the :ref:`eccentric anomaly <kepE_ad>`.
The final result, for :math:`n > 0`, is:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = \frac{1}{n \left(1 - c^{\left[ 0 \right]}\left( t \right) -d^{\left[ 0 \right]}\left( t \right)\right)}
   \left\{
   n\left( k^{\left[ n \right]}\left( t \right) e^{\left[ 0 \right]}\left( t \right)
   - h^{\left[ n \right]}\left( t \right) f^{\left[ 0 \right]}\left( t \right)
   + \lambda^{\left[ n \right]}\left( t \right)\right) +
   \sum_{j=1}^{n-1}j\left[
      a^{\left[ j \right]}\left( t \right) \left(
         c^{\left[ n - j \right]}\left( t \right) + d^{\left[ n - j \right]}\left( t \right)
      \right)
      + k^{\left[ j \right]}\left( t \right) e^{\left[ n - j \right]}\left( t \right)
      - h^{\left[ j \right]}\left( t \right) f^{\left[ n - j \right]}\left( t \right)
   \right]
   \right\}.

Time functions
--------------

Time polynomials
^^^^^^^^^^^^^^^^

Given the time polynomial of order :math:`n`

.. math::
   :label:

   p_n\left( t \right) = \sum_{i=0}^n a_i t^i,

its derivative of order :math:`j` is

.. math::
   :label:

   \left(p_n\left( t \right)\right)^{\left( j \right)} = \sum_{i=j}^n \left( i \right)_j a_i t^{i - j},

where :math:`\left( i \right)_j` is the `falling factorial <https://en.wikipedia.org/wiki/Falling_and_rising_factorials>`__.
The normalised derivative of order :math:`j` is

.. math::
   :label:

   \left(p_n\left( t \right)\right)^{\left[ j \right]} = \frac{1}{j!}\sum_{i=j}^n \left( i \right)_j a_i t^{i - j},

which, with the help of elementary relations involving factorials and after re-arranging the indices, can be rewritten as

.. math::
   :label:

   \left(p_n\left( t \right)\right)^{\left[ j \right]} = \sum_{i=0}^{n-j} {i+j \choose j} a_{i+j} t^i.
