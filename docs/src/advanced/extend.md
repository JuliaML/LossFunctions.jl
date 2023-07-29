```@meta
DocTestSetup = quote
    using LossFunctions
    using LossFunctions.Traits
end
```

# Altering existing Losses

There are situations in which one wants to work with slightly
altered versions of specific loss functions. This package
provides two generic ways to create such meta losses for specific
families of loss functions.

1. Scaling a supervised loss by a constant real number. This is
   done at compile time and can in some situations even lead to
   simpler code (e.g. in the case of the derivative for a
   [`L2DistLoss`](@ref))

2. Weighting the classes of a margin-based loss differently in
   order to better deal with unbalanced binary classification
   problems.

## Scaling a Supervised Loss

It is quite common in machine learning courses to define the
least squares loss as ``\frac{1}{2} (\hat{y} - y)^2``, while this
package implements that type of loss as an ``L_2`` distance loss
using ``(\hat{y} - y)^2``, i.e. without the constant scale
factor.

For situations in which one wants a scaled version of an existing
loss type, we provide the concept of a **scaled loss**. The
difference is literally only a constant real number that gets
multiplied to the existing implementation of the loss function
(and derivatives).

```@docs
ScaledLoss
```

```jldoctest
julia> lsloss = 1/2 * L2DistLoss()
ScaledLoss{L2DistLoss, 0.5}(L2DistLoss())

julia> L2DistLoss()(4.0, 0.0)
16.0

julia> lsloss(4.0, 0.0)
8.0
```

As you have probably noticed, the constant scale factor gets
promoted to a type-parameter. This can be quite an overhead when
done on the fly every time the loss value is computed. To avoid
this one can make use of `Val` to specify the scale factor in a
type-stable manner.

```jldoctest
julia> sl = ScaledLoss(L2DistLoss(), Val(0.5))
ScaledLoss{L2DistLoss, 0.5}(L2DistLoss())
```

Storing the scale factor as a type-parameter instead of a member
variable has some nice advantages. It allows the compiler to do
some quite convenient optimizations if possible. For example the
compiler is able to figure out that the derivative simplifies for
a scaled loss. This is accomplished using the power of `@fastmath`.

## Reweighting a Margin Loss

It is not uncommon in classification scenarios to find yourself
working with in-balanced data sets, where one class has much more
observations than the other one. There are different strategies
to deal with this kind of problem. The approach that this package
provides is to weight the loss for the classes differently. This
basically means that we penalize mistakes in one class more than
mistakes in the other class. More specifically we scale the loss
of the positive class by the weight-factor ``w`` and the loss
of the negative class with ``1-w``.

```julia-repl
if target > 0
    w * loss(target, output)
else
    (1-w) * loss(target, output)
end
```


Instead of providing special functions to compute a class-weighted loss,
we instead expose a generic way to create new weighted versions of already
existing unweighted margin losses. This way, every existing subtype of
[`MarginLoss`](@ref) can be re-weighted arbitrarily. Furthermore, it
allows every algorithm that expects a binary loss to work with weighted
binary losses as well.

```@docs
WeightedMarginLoss
```

```jldoctest weighted
julia> myloss = WeightedMarginLoss(HingeLoss(), 0.8)
WeightedMarginLoss{L1HingeLoss, 0.8}(L1HingeLoss())

julia> myloss(-4.0, 1.0) # positive class
4.0

julia> HingeLoss()(-4.0, 1.0)
5.0

julia> myloss(4.0, -1.0) # negative class
0.9999999999999998

julia> HingeLoss()(4.0, -1.0)
5.0
```

Note that the scaled version of a margin-based loss does not
anymore belong to the family of margin-based losses itself. In
other words the resulting loss is neither a subtype of
[`MarginLoss`](@ref), nor of the original type of loss.

```jldoctest weighted
julia> typeof(myloss) <: MarginLoss
false

julia> typeof(myloss) <: HingeLoss
false
```

Similar to scaled losses, the constant weight factor gets
promoted to a type-parameter. This can be quite an overhead when
done on the fly every time the loss value is computed. To avoid
this one can make use of `Val` to specify the scale factor in a
type-stable manner.

```jldoctest weighted
julia> WeightedMarginLoss(HingeLoss(), Val(0.8))
WeightedMarginLoss{L1HingeLoss, 0.8}(L1HingeLoss())
```
