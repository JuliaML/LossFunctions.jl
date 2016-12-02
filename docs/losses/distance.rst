Distance-based Losses
=====================

This section lists all the subtypes of :class:`DistanceLoss`
that are implemented in this package.

.. figure:: https://rawgithub.com/JuliaML/FileStorage/master/LossFunctions/distance.svg

   Distance-based Losses (Regression)

LPDistLoss
-----------

.. class:: LPDistLoss

   The :math:`p`-th power absolute distance loss.
   It is Lipschitz continuous iff :math:`p = 1`, convex if and only
   if :math:`p \ge 1`, and strictly convex iff :math:`p > 1`.

+----------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------+
| Lossfunction                                                                           | Derivative                                                                             |
+========================================================================================+========================================================================================+
| .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/LPDistLoss1.svg | .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/LPDistLoss2.svg |
+----------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------+
| .. math:: L(r) = | r | ^p                                                              | .. math:: L'(r) = p \cdot r \cdot | r | ^{p-2}                                         |
+----------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------+


L1DistLoss
-----------

+----------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------+
| Lossfunction                                                                           | Derivative                                                                             |
+========================================================================================+========================================================================================+
| .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L1DistLoss1.svg | .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L1DistLoss2.svg |
+----------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------+

.. class:: L1DistLoss

   The absolute distance loss. Special case of the :class:`LPDistLoss`
   with ``P=1``.
   It is Lipshitz continuous and convex, but not strictly convex.

.. math::

   L(r) = | r |

L2DistLoss
-----------

+----------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------+
| Lossfunction                                                                           | Derivative                                                                             |
+========================================================================================+========================================================================================+
| .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L2DistLoss1.svg | .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L2DistLoss2.svg |
+----------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------+

.. class:: L2DistLoss

   The least squares loss. Special case of the :class:`LPDistLoss`
   with ``P=2``. It is strictly convex.

.. math::

   L(r) = | r | ^2

LogitDistLoss
--------------

+-------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| Lossfunction                                                                              | Derivative                                                                                |
+===========================================================================================+===========================================================================================+
| .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/LogitDistLoss1.svg | .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/LogitDistLoss2.svg |
+-------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+

.. class:: LogitDistLoss

   The distance-based logistic loss for regression.
   It is strictly convex and Lipshitz continuous.

.. math::

   L(r) = - \ln \frac{4 e^r}{(1 + e^r)^2}

HuberLoss
-----------

+----------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------+
| Lossfunction                                                                           | Derivative                                                                             |
+========================================================================================+========================================================================================+
| .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/HuberLoss1.svg  | .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/HuberLoss2.svg  |
+----------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------+

.. class:: HuberLoss

   .. attribute:: α

   Loss function commonly used for robustness to outliers.
   For large values of :math:`\alpha` it becomes close to the
   :class:`L1DistLoss`, while for small values of :math:`\alpha`
   it resembles the :class:`L2DistLoss`.
   It is Lipshitz continuous and convex, but not strictly convex.

.. math::

   L(r) = \begin{cases} \frac{r^2}{2} & \quad \text{if } | r | \le \alpha \\ \alpha | r | - \frac{\alpha^2}{2} & \quad \text{otherwise}\\ \end{cases}

L1EpsilonInsLoss
-----------------

+-----------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Lossfunction                                                                                  | Derivative                                                                                    |
+===============================================================================================+===============================================================================================+
| .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L1EpsilonInsLoss1.svg  | .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L1EpsilonInsLoss2.svg  |
+-----------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+

.. class:: L1EpsilonInsLoss

   .. attribute:: ϵ

   The :math:`\epsilon`-insensitive loss. Typically used in linear
   support vector regression. It ignores deviances smaller than
   :math:`\epsilon` , but penalizes larger deviances linarily.
   It is Lipshitz continuous and convex, but not strictly convex.

.. math::

   L(r) = \max \{ 0, | r | - \epsilon \}

L2EpsilonInsLoss
-----------------

+-----------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Lossfunction                                                                                  | Derivative                                                                                    |
+===============================================================================================+===============================================================================================+
| .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L2EpsilonInsLoss1.svg  | .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L2EpsilonInsLoss2.svg  |
+-----------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+

.. class:: L2EpsilonInsLoss

   .. attribute:: ϵ

   The :math:`\epsilon`-insensitive loss. Typically used in linear
   support vector regression. It ignores deviances smaller than
   :math:`\epsilon` , but penalizes larger deviances quadratically.
   It is convex, but not strictly convex.

.. math::

   L(r) = \max \{ 0, | r | - \epsilon \}^2

PeriodicLoss
-------------

+-----------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Lossfunction                                                                                  | Derivative                                                                                    |
+===============================================================================================+===============================================================================================+
| .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/PeriodicLoss1.svg      | .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/PeriodicLoss2.svg      |
+-----------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+

.. class:: PeriodicLoss

   .. attribute:: c

   Measures distance on a circle of specified circumference :math:`c`.

.. math::

   L(r) = 1 - \cos \left ( \frac{2 r \pi}{c} \right )

QuantileLoss
-------------

+-----------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Lossfunction                                                                                  | Derivative                                                                                    |
+===============================================================================================+===============================================================================================+
| .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/QuantileLoss1.svg      | .. image:: https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/QuantileLoss2.svg      |
+-----------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+

.. class:: QuantileLoss

   .. attribute:: τ

    The quantile loss, aka pinball loss. Typically used to estimate
    the conditional :math:`\tau`-quantiles.
    It is convex, but not strictly convex. Furthermore it is
    Lipschitz continuous.

.. math::

   L(r) = \begin{cases} -\left( 1 - \tau  \right) r & \quad \text{if } r < 0 \\ \tau r & \quad \text{if } r \ge 0 \\ \end{cases}

