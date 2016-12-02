Margin-based Losses
====================

This section lists all the subtypes of :class:`MarginLoss`
that are implemented in this package.

.. figure:: https://rawgithub.com/JuliaML/FileStorage/master/LossFunctions/margin.svg

   Margin-based Losses (Classification)

ZeroOneLoss
------------

.. class:: ZeroOneLoss

   The classical classification loss. It penalizes every
   missclassified observation with a loss of `1` while every
   correctly classified observation has a loss of `0`.
   It is not convex nor continuous and thus seldomly used directly.
   Instead one usually works with some classification-calibrated
   surrogate loss, such as one of those listed below.

.. math::

   L(a) = \begin{cases} 1 & \quad \text{if } a < 0 \\ 0 & \quad \text{if } a >= 0\\ \end{cases}

PerceptronLoss
---------------

.. class:: PerceptronLoss

   The perceptron loss linearly penalizes every prediction where the
   resulting ``agreement`` :math:`a \le 0`.
   It is Lipshitz continuous and convex, but not strictly convex.

.. math::

   L(a) = \max \{ 0, - a \}

L1HingeLoss
------------

.. class:: L1HingeLoss

   The hinge loss linearly penalizes every predicition where the
   resulting ``agreement`` :math:`a \le 1` .
   It is Lipshitz continuous and convex, but not strictly convex.

.. math::

   L(a) = \max \{ 0, 1 - a \}


L2HingeLoss
------------

.. class:: L2HingeLoss

   The truncated least squares loss quadratically penalizes every
   predicition where the resulting ``agreement`` :math:`a \le 1` .
   It is locally Lipshitz continuous and convex,
   but not strictly convex.

.. math::

   L(a) = \max \{ 0, 1 - a \} ^2

LogitMarginLoss
----------------

.. class:: LogitMarginLoss

   The margin version of the logistic loss. It is infinitely many
   times differentiable, strictly convex, and lipschitz continuous.

.. math::

   L(a) = \ln (1 + e^{-a})

SmoothedL1HingeLoss
---------------------

.. class:: SmoothedL1HingeLoss

   .. attribute:: γ

   As the name suggests a smoothed version of the L1 hinge loss.
   It is Lipshitz continuous and convex, but not strictly convex.

.. math::

   L(a) = \begin{cases} \frac{0.5}{\gamma} \cdot \max \{ 0, 1 - a \} ^2 & \quad \text{if } a \ge 1 - \gamma \\ 1 - \frac{\gamma}{2} - a & \quad \text{otherwise}\\ \end{cases}

ModifiedHuberLoss
-------------------

.. class:: ModifiedHuberLoss

   A special (scaled) case of the :class:`SmoothedL1HingeLoss` with
   :math:`\gamma = 4`.
   It is Lipshitz continuous and convex, but not strictly convex.

.. math::

   L(a) = \begin{cases} \max \{ 0, 1 - a \} ^2 & \quad \text{if } a \ge -1 \\ - 4 a & \quad \text{otherwise}\\ \end{cases}


L2MarginLoss
-------------

.. class:: L2MarginLoss

   The margin-based least-squares loss for classification, which
   quadratically penalizes every prediction where :math:`a \ne 1`.
   It is locally Lipschitz continuous and strongly convex.

.. math::

   L(a) = {\left( 1 - a \right)}^2

ExpLoss
--------

.. class:: ExpLoss

   The margin-based exponential Loss used for classification,
   which penalizes every prediction exponentially. It is
   infinitely many times differentiable, locally Lipschitz
   continuous and strictly convex, but not clipable.

.. math::

   L(a) = e^{-a}

SigmoidLoss
------------

.. class:: SigmoidLoss

   The so called sigmoid loss is a continuous margin-base loss
   which penalizes every prediction with a loss within in the
   range (0,2). It is infinitely many times differentiable,
   Lipschitz continuous but nonconvex.

.. math::

   L(a) = 1 - \tanh(a)

DWDMarginLoss
-------------

.. class:: DWDMarginLoss

   .. attribute:: q

   The distance weighted discrimination margin loss.
   A differentiable generalization of the L1 hinge loss that is
   different than the :class:`SmoothedL1HingeLoss`

.. math::

   L(a) = \begin{cases} 1 - a & \quad \text{if } a \ge \frac{q}{q+1} \\ \frac{1}{a^q} \frac{q^q}{(q+1)^{q+1}} & \quad \text{otherwise}\\ \end{cases}

