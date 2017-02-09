Altering existing Losses
=================================

There are situations in which one wants to work with slightly
altered versions of specific loss functions. This package
provides two generic ways to create such meta losses for specific
families of loss functions.

1. Scaling a supervised loss by a constant real number. This is
   done at compile time and can in some situations even lead to
   simpler code (e.g. in the case of the derivative for a
   :class:`L2DistLoss`)

2. Weighting the classes of a margin-based loss differently in
   order to better deal with unbalanced binary classification
   problems.

Scaling a Supervised Loss
----------------------------

It is quite common in machine learning courses to define the
least squares loss as :math:`\frac{1}{2} (\hat{y} - y)^2`, while
this package implements that type of loss as an :math:`L_2`
distance loss using :math:`(\hat{y} - y)^2`, i.e. without the
constant scale factor.

For situations in which one wants a scaled version of an existing
loss type, we provide the concept of a **scaled loss**. The
difference is literally only a constant real number that gets
multiplied to the existing implementation of the loss function
(and derivatives).

.. code-block:: jlcon

   julia> lsloss = 1/2 * L2DistLoss()
   LossFunctions.ScaledDistanceLoss{LossFunctions.LPDistLoss{2},0.5}(LossFunctions.LPDistLoss{2}())

   julia> value(L2DistLoss(), 0.0, 4.0)
   16.0

   julia> value(lsloss, 0.0, 4.0)
   8.0

While the resulting loss is of the same basic family as the
original loss (i.e. margin-based or distance-based), it is not a
sub-type of it.

.. code-block:: jlcon

   julia> typeof(lsloss) <: DistanceLoss
   true

   julia> typeof(lsloss) <: L2DistLoss
   false

As you have probably noticed, the constant scale factor gets
promoted to a type-parameter. This can be quite an overhead when
done on the fly every time the loss value is computed. To avoid
this one can make use of ``Val`` to specify the scale factor in a
type-stable manner.

.. code-block:: jlcon

   julia> lsloss = scaledloss(L2DistLoss(), Val{0.5})
   LossFunctions.ScaledDistanceLoss{LossFunctions.LPDistLoss{2},0.5}(LossFunctions.LPDistLoss{2}())

Storing the scale factor as a type-parameter instead of a member
variable has some nice advantages. For one it makes it possible
to define new types of losses using simple type-aliases.

.. code-block:: jlcon

   julia> typealias LeastSquaresLoss LossFunctions.ScaledDistanceLoss{L2DistLoss,0.5}
   LossFunctions.ScaledDistanceLoss{LossFunctions.LPDistLoss{2},0.5}

   julia> value(LeastSquaresLoss(), 0.0, 4.0)
   8.0

Furthermore, it allows the compiler to do some quite convenient
optimizations if possible. For example the compiler is able to
figure out that the derivative simplifies for our newly defined
``LeastSquaresLoss``, because ``1/2 * 2`` cancels each other.
This is accomplished using the power of ``@fastmath``.

.. code-block:: jlcon

   julia> @code_llvm deriv(L2DistLoss(), 0.0, 4.0)
   define double @julia_deriv_71652(double, double) #0 {
   top:
     %2 = fsub double %1, %0
     %3 = fmul double %2, 2.000000e+00
     ret double %3
   }

   julia> @code_llvm deriv(LeastSquaresLoss(), 0.0, 4.0)
   define double @julia_deriv_71659(double, double) #0 {
   top:
     %2 = fsub double %1, %0
     ret double %2
   }

Reweighting a Margin Loss
----------------------------

It is not uncommon in classification scenarios to find yourself
working with in-balanced data sets, where one class has much more
observations than the other one. There are different strategies
to deal with this kind of problem. The approach that this package
provides is to weight the loss for the classes differently. This
basically means that we penalize mistakes in one class more than
mistakes in the other class. More specifically we scale the loss
of the positive class by the weight-factor :math:`w` and the loss
of the negative class with :math:`1-w`.

.. code-block:: julia

   if target > 0
       w * loss(target, output)
   else
       (1-w) * loss(target, output)
   end


Instead of providing special functions to compute a
class-weighted loss, we instead expose a generic way to create
new weighted versions of already existing unweighted losses. This
way, every existing subtype of :class:`MarginLoss` can be
re-weighted arbitrarily. Furthermore, it allows every algorithm
that expects a binary loss to work with weighted binary losses as
well.

.. code-block:: jlcon

   julia> myloss = weightedloss(HingeLoss(), 0.8)
   LossFunctions.WeightedBinaryLoss{LossFunctions.L1HingeLoss,0.8}(LossFunctions.L1HingeLoss())

   # positive class
   julia> value(myloss, 1.0, -4.0)
   4.0

   julia> value(HingeLoss(), 1.0, -4.0)
   5.0

   # negative class
   julia> value(myloss, -1.0, 4.0)
   1.0

   julia> value(HingeLoss(), -1.0, 4.0)
   5.0

Note that the scaled version of a margin-based loss does not
anymore belong to the family of margin-based losses itself. In
other words the resulting loss is neither a subtype of
:class:`MarginLoss`, nor of the original type of loss.

.. code-block:: jlcon

   julia> typeof(myloss) <: MarginLoss
   false

   julia> typeof(myloss) <: HingeLoss
   false

Similar to scaled losses, the constant weight factor gets
promoted to a type-parameter. This can be quite an overhead when
done on the fly every time the loss value is computed. To avoid
this one can make use of ``Val`` to specify the scale factor in a
type-stable manner.

.. code-block:: jlcon

   julia> myloss = weightedloss(HingeLoss(), Val{0.8})
   LossFunctions.WeightedBinaryLoss{LossFunctions.L1HingeLoss,0.8}(LossFunctions.L1HingeLoss())

Storing the scale factor as a type-parameter instead of a member
variable has a nice advantage. It makes it possible to define new
types of losses using simple type-aliases.

.. code-block:: jlcon

   julia> typealias MyWeightedHingeLoss LossFunctions.WeightedBinaryLoss{HingeLoss,0.8}
   LossFunctions.WeightedBinaryLoss{LossFunctions.L1HingeLoss,0.8}

   julia> value(MyWeightedHingeLoss(), 1.0, -4.0)
   4.0

