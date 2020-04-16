"""
    WeightedBinaryLoss{L,W} <: MarginLoss

Can an be used to represent a re-weighted version of some type of
binary loss `L`. The weight-factor `W`, which must be in `[0, 1]`,
denotes the relative weight of the positive class, while the
relative weight of the negative class will be `1 - W`.
For example: To create a typealias for a 1:4 weighted version of
`L2HingeLoss`, type:

```julia
const WeightedL2HingeLoss = LossFunctions.WeightedBinaryLoss{L2HingeLoss,0.2}
```

This new loss-type can then be instantiated in the same manner and
with the same parameters as the original unscaled loss-type.

In contrast, in order to only create a re-weighted instance of some
specific loss you can use `weightedloss(L2HingeLoss(), Val(0.2))`.
See `?weightedloss` for more information.
"""
struct WeightedBinaryLoss{L<:MarginLoss,W} <: MarginLoss
    loss::L
    WeightedBinaryLoss{L,W}(loss::L) where {L<:MarginLoss, W} = new{L,W}(loss)
end

@generated function (::Type{WeightedBinaryLoss{L,W}})(args...) where {L<:MarginLoss, W}
    typeof(W) <: Number && 0 <= W <= 1 || _werror()
    :(WeightedBinaryLoss{L,W}(L(args...)))
end

_werror() = throw(ArgumentError("The given \"weight\" has to be a number in the interval [0, 1]"))

@generated function WeightedBinaryLoss(loss::L, ::Val{W}) where {L<:MarginLoss,W}
    typeof(W) <: Number && 0 <= W <= 1 || _werror()
    :(WeightedBinaryLoss{L,W}(loss))
end

function WeightedBinaryLoss(loss::MarginLoss, w::Number)
    WeightedBinaryLoss(loss, Val(w))
end

@generated function WeightedBinaryLoss(s::WeightedBinaryLoss{T,W1}, ::Val{W2}) where {T,W1,W2}
    :(WeightedBinaryLoss(s.loss, Val($(W1*W2))))
end

for fun in (:value, :deriv, :deriv2)
    @eval function ($fun)(l::WeightedBinaryLoss{L,W}, target::Number, output::Number) where {L,W}
        # We interpret the W to be the weight of the positive class
        if target > 0
            W * ($fun)(l.loss, target, output)
        else
            (1-W) * ($fun)(l.loss, target, output)
        end
    end
end

# An α-weighted version of a classification callibrated margin loss is
# itself classification callibrated if and only if α == 1/2
isclasscalibrated(l::WeightedBinaryLoss{T,W}) where {T,W} = W == 0.5 && isclasscalibrated(l.loss)

# TODO: Think about this semantic
issymmetric(::WeightedBinaryLoss) = false

for prop in [:isminimizable, :isdifferentiable,
             :istwicedifferentiable,
             :isconvex, :isstrictlyconvex,
             :isstronglyconvex, :isnemitski,
             :isunivfishercons, :isfishercons,
             :islipschitzcont, :islocallylipschitzcont,
             :isclipable, :ismarginbased,
             :isdistancebased]
    @eval ($prop)(l::WeightedBinaryLoss) = ($prop)(l.loss)
end

for prop_param in (:isdifferentiable, :istwicedifferentiable)
    @eval ($prop_param)(l::WeightedBinaryLoss, at) = ($prop_param)(l.loss, at)
end
