Getting Started
================

This section outlines the basic steps needed to start utilizing
the LossFunctions.jl package.
To that end we will provide a condensed overview of the package.

.. note::

    In order to keep this overview consise, we will not discuss any
    background information or theory on the losses here in detail.

Installation
--------------

To install LossFunctions.jl, start up Julia and type the following
code-snipped into the REPL. It makes use of the native Julia
package manger.

.. code-block:: julia

    Pkg.add("LossFunctions")

Additionally, for example if you encounter any sudden issues,
or in the case you would like to contribute to the package,
you can manually choose to be on the latest (untagged) version.

.. code-block:: julia

    Pkg.checkout("LossFunctions")


Hello World
------------

This package is registered in the Julia package ecosystem. Once
installed the package can be imported just as any other Julia
package.

.. code-block:: julia

    using LossFunctions

The following code snippets show a simple scenario of how a
`Loss` can be used to compute the element-wise values.

.. code-block:: julia

    using LossFunctions

    true_targets = [  1,  0, -2]
    pred_outputs = [0.5,  1, -1]

    value(L2DistLoss(), true_targets, pred_outputs)

.. code-block:: none

    3-element Array{Float64,1}:
     0.25
     1.0
     1.0

Alternatively, one can also use the loss like a function to
compute its :func:`value`.

.. code-block:: julia

    myloss = L2DistLoss()
    myloss(true_targets, pred_outputs) # same result as above

The function signatures of :func:`value` also apply to the derivatives.

.. code-block:: julia

    deriv(L2DistLoss(), true_targets, pred_outputs)

.. code-block:: none

    3-element Array{Float64,1}:
     -1.0
     2.0
     2.0

Additionally, we provide mutating versions of most functions.

.. code-block:: julia

    buffer = zeros(3)
    deriv!(buffer, L2DistLoss(), true_targets, pred_outputs)

If need be, one can also compute the :func:`meanvalue` or
:func:`sumvalue` efficiently, without allocating a temporary array.

.. code-block:: julia

    # or meanvalue
    sumvalue(L2DistLoss(), true_targets, pred_outputs)

.. code-block:: none

    0.75


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
    `issue #9 <https://github.com/JuliaML/LossFunctions.jl/issues/9#issuecomment-190321549>`_

Margin-based losses are usually used for binary classification.
In contrast to other formalism, they do not natively provide
probabilities as output.

.. note::

    Even though distance-based losses and margin-based losses
    can be expressed in univariate form, we still provide the
    bivariate form of ``value`` for both.


Getting Help
-------------

To get help on specific functionality you can either look up the
information here, or if you prefer you can make use of Julia's
native doc-system.
The following example shows how to get additional information
on :class:`L1HingeLoss` within Julia's REPL:

.. code-block:: julia

    ?L1HingeLoss

.. code-block:: none

   search: L1HingeLoss SmoothedL1HingeLoss

     L1HingeLoss <: MarginLoss

      The hinge loss linearly penalizes every predicition where the resulting
      agreement <= 1 . It is Lipschitz continuous and convex, but not strictly
      convex.

    L(y, ŷ) = max(0, 1 - y⋅ŷ)

                Lossfunction                     Derivative
        ┌────────────┬────────────┐      ┌────────────┬────────────┐
      3 │'\.                      │    0 │                  ┌------│
        │  ''_                    │      │                  |      │
        │     \.                  │      │                  |      │
        │       '.                │      │                  |      │
      L │         ''_             │   L' │                  |      │
        │            \.           │      │                  |      │
        │              '.         │      │                  |      │
      0 │                ''_______│   -1 │------------------┘      │
        └────────────┴────────────┘      └────────────┴────────────┘
        -2                        2      -2                        2
                   y ⋅ ŷ                            y ⋅ ŷ


If you find yourself stuck or have other questions concerning the
package you can find us at gitter or the *Machine Learning*
domain on discourse.julialang.org

- `Julia ML on Gitter <https://gitter.im/JuliaML/chat>`_

- `Machine Learning on Julialang <https://discourse.julialang.org/c/domain/ML>`_

If you encounter a bug or would like to participate in the
further development of this package come find us on Github.

- `JuliaML/LossFunctions.jl <https://github.com/JuliaML/LossFunctions.jl>`_

