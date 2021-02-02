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
