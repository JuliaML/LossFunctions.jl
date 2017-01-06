Getting Started
================

LossFunctions is the result of a collaborative effort to design
an efficient but also convenient-to-use `Julia
<http://julialang.org/>`_ library that provides the most commonly
utilized loss functions in Machine Learning. As such, this
package implements the functionality needed to query various
properties about a loss (e.g. convexity), as well as a number of
methods to compute its value, derivative, and second derivative
for single observations or arrays of observations.

In this section we will provide a condensed overview of the
package. In order to keep this overview concise, we will not
discuss any background information or theory on the losses here
in detail.

Installation
--------------

To install `LossFunctions.jl
<https://github.com/JuliaML/LossFunctions.jl>`_, start up Julia
and type the following code-snipped into the REPL. It makes use
of the native Julia package manger.

.. code-block:: julia

   Pkg.add("LossFunctions")

Additionally, for example if you encounter any sudden issues,
or in the case you would like to contribute to the package,
you can manually choose to be on the latest (untagged) version.

.. code-block:: julia

   Pkg.checkout("LossFunctions")

Overview
------------

Let us take a look at a few examples to get a feeling of how one
can use this library. This package is registered in the Julia
package ecosystem. Once installed the package can be imported
as usual.

.. code-block:: julia

   using LossFunctions

Typically the losses we work with in Machine Learning are
multivariate functions of the **true target** :math:`y`, which
represents the "ground truth" (i.e. correct answer), and the
**predicted output** :math:`\hat{y}`, which is what our model
thinks the truth is. All losses that can be expressed this way
will be referred to as supervised losses. The true targets are
often expected to be of a specific set (e.g. in classification),
which will refer to as :math:`Y`, while the predicted outputs may
be any real number. A supervised loss is thus for our purposes
defined as:

.. math::

   L : Y \times \mathbb{R} \rightarrow [0,\infty)

Such a loss function takes these two variables and returns us a
value that quantifies how "bad" our prediction is when comparing
it to the truth. In other words: the lower the loss the better
the prediction.

From an implementation perspective we should point out that all
the concrete loss "functions" that this package provides, are
actually defined as immutable types instead of native Julia
functions. To then compute the value of some loss we provide the
function :func:`value`. To start off, note that at its core all
the provided losses of this package are defined on single
variables (i.e. numbers).

.. code-block:: jlcon

   #                loss       y    ŷ
   julia> value(L2DistLoss(), 1.0, 0.5)
   0.25

Calling the same function using arrays instead of numbers will
default to returning the element-wise results, and thus basically
just serve as a wrapper for broadcast.

.. code-block:: jlcon

   julia> true_targets = [  1,  0, -2];

   julia> pred_outputs = [0.5,  2, -1];

   julia> value(L2DistLoss(), true_targets, pred_outputs)
   3-element Array{Float64,1}:
    0.25
    4.0
    1.0

Alternatively, one can also use an instance of a loss just like
one would use any other Julia function. This can make the code
significantly more readable while not impacting performance as it
is a zero-cost abstraction (i.e. it compiles down to the same
code).

.. code-block:: jlcon

   julia> loss = L2DistLoss()
   LossFunctions.LPDistLoss{2}()

   julia> loss(true_targets, pred_outputs) # same result as above
   3-element Array{Float64,1}:
    0.25
    4.0
    1.0

   julia> loss(1, 0.5f0) # single observation
   0.25f0

If you are not actually interested in the element-wise results,
but some accumulation of those (such as mean or sum), you can
additionally specify an **average mode**. This will avoid
allocating a temporary array and directly compute the result.

.. code-block:: jlcon

   julia> value(L2DistLoss(), true_targets, pred_outputs, AvgMode.Sum())
   5.25

   julia> value(L2DistLoss(), true_targets, pred_outputs, AvgMode.Mean())
   1.75

Aside from these standard unweighted average modes, we also
provide weighted alternatives.

.. code-block:: jlcon

   julia> value(L2DistLoss(), true_targets, pred_outputs, AvgMode.WeightedSum([2,1,1]))
   5.5

   julia> value(L2DistLoss(), true_targets, pred_outputs, AvgMode.WeightedMean([2,1,1]))
   1.375

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



We can further divide the supervised losses into two useful
sub-categories: :class:`DistanceLoss` for regression and
:class:`MarginLoss` for classification.

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

