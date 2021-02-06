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

Given :math:`a\left( t \right) = \cosh b\left( t \right)`, the process of deriving of :math:`a^{\left[ n \right]}\left( t \right)` is
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

Given :math:`a\left( t \right) = \tanh b\left( t \right)`, the process of deriving of :math:`a^{\left[ n \right]}\left( t \right)` is
identical to the :ref:`tangent <ad_tan>`, apart from a sign change. After the definition of the auxiliary function

.. math::
   :label:

   c\left( t \right)  = a^2\left( t \right) ,

the final result, for :math:`n > 0`, is:

.. math::
   :label:

   a^{\left[ n \right]}\left( t \right) = b^{\left[ n \right]}\left( t \right) - \frac{1}{n}\sum_{j=1}^{n} j c^{\left[ n - j \right]}\left( t \right) b^{\left[ j \right]}\left( t \right).
