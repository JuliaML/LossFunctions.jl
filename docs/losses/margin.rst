Margin-based Losses
====================

This section lists all the subtypes of :class:`MarginLoss`
that are implemented in this package.

.. figure:: https://cloud.githubusercontent.com/assets/10854026/17837728/62da0bac-67bb-11e6-92eb-fd5b291cdd8a.png

   Margin-based Losses (Classification)

   Note: The ZeroOneLoss itself is not margin-based

PerceptronLoss
---------------

.. class:: PerceptronLoss

   The perceptron loss linearly penalizes every prediction where the
   resulting ``agreement`` :math:`a \le 0``
   It is Lipshitz continuous and convex, but not stricktly convex.

.. math::

   L(a) = \max \{ 0, - a \}

L1HingeLoss
------------

.. class:: L1HingeLoss

   The hinge loss linearly penalizes every predicition where the
   resulting ``agreement`` :math:`a \le 1` .
   It is Lipshitz continuous and convex, but not stricktly convex.

.. math::

   L(a) = \max \{ 0, 1 - a \}


L2HingeLoss
------------

.. class:: L2HingeLoss

   The truncated least squares loss quadratically penalizes every
   predicition where the resulting ``agreement`` :math:`a \le 1` .
   It is locally Lipshitz continuous and convex,
   but not stricktly convex.

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
   It is Lipshitz continuous and convex, but not stricktly convex.

.. math::

   L(a) = \begin{cases} \frac{0.5}{\gamma} \cdot \max \{ 0, 1 - a \} ^2 & \quad \text{if } a \ge 1 - \gamma \\ 1 - \frac{\gamma}{2} - a & \quad \text{otherwise}\\ \end{cases}

ModifiedHuberLoss
-------------------

.. class:: ModifiedHuberLoss

   A special (scaled) case of the :class:`SmoothedL1HingeLoss` with
   :math:`\gamma = 4`.
   It is Lipshitz continuous and convex, but not stricktly convex.

.. math::

   L(a) = \begin{cases} \max \{ 0, 1 - a \} ^2 & \quad \text{if } a \ge -1 \\ - 4 a & \quad \text{otherwise}\\ \end{cases}


