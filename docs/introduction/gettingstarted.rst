Getting Started
================

This section outlines the basic steps needed to start utilizing
the Losses.jl package.
To that end we will provide a condensed overview of the package.

.. note::

    In order to keep this overview consise, we will not discuss any
    background information or theory on the losses here in detail.

Overview
---------

All the concrete loss "functions" that this package provides are
defined as types and are subtypes of the abstract ``Loss``.

Typically the losses we work with in Machine Learning are bivariate
functions of the true ``target`` and the predicted ``output`` of
some prediction model. All losses that can be expressed this way
are subtypes for :class:`SupervisedLoss`.
To compute the value of some :class:`SupervisedLoss` we use the
function :func:`value`.

.. code-block:: julia

    value(L2DistLoss(), true_target, pred_output)

We can further divide the supervised losses into two useful
sub-categories: :class:`DistanceLoss` and :class:`MarginLoss`.


Losses for Regression
~~~~~~~~~~~~~~~~~~~~~~

Supervised losses that can be expressed as a univariate function
of ``output - target`` are referred to as distance-based losses.

.. code-block:: julia

    value(L2DistLoss(), difference)

Distance-based losses are typically utilized for regression problems.
That said, there are also other losses that are useful for
regression problems that don't fall into this category, such as
the :class:`PeriodicLoss`.

.. note::

    In the literature that this package is partially based on,
    the convention for the distance-based losses is ``target - output``
    (see [STEINWART2008]_ p. 38).
    We chose to diverge from this definition because that would
    cause the the sign of the derivative to flip.

Losses for Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Margin-base losses are supervised losses where the values of
the targets are restricted to be in {-1, 1}, and which can be
expressed as a univariate function ``output * target``.

.. code-block:: julia

    value(L1HingeLoss(), agreement)

.. note::

    Throughout the codebase we refer to the result of
    ``output * target`` as ``agreement``.
    The discussion that lead to this convention can be found
    `issue #9 <https://github.com/JuliaML/Losses.jl/issues/9#issuecomment-190321549>`_

Margin-based losses are usually used for binary classification.
In contrast to other formalism, they do not natively provide
probabilities as output.

.. note::

    Even though distance-based losses and margin-based losses
    can be expressed in univariate form, we still provide the
    bivariate form of ``value`` for both.



