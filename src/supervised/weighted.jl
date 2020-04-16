"""
    WeightedMarginLoss{L,W} <: MarginLoss

Can an be used to represent a re-weighted version of some type of
binary loss `L`. The weight-factor `W`, which must be in `[0, 1]`,
denotes the relative weight of the positive class, while the
relative weight of the negative class will be `1 - W`.
"""
struct WeightedMarginLoss{L<:MarginLoss,W} <: MarginLoss
    loss::L
    WeightedMarginLoss{L,W}(loss::L) where {L<:MarginLoss, W} = new{L,W}(loss)
end

@generated function (::Type{WeightedMarginLoss{L,W}})(args...) where {L<:MarginLoss, W}
    typeof(W) <: Number && 0 <= W <= 1 || _werror()
    :(WeightedMarginLoss{L,W}(L(args...)))
end

_werror() = throw(ArgumentError("The given \"weight\" has to be a number in the interval [0, 1]"))

@generated function WeightedMarginLoss(loss::L, ::Val{W}) where {L<:MarginLoss,W}
    typeof(W) <: Number && 0 <= W <= 1 || _werror()
    :(WeightedMarginLoss{L,W}(loss))
end

function WeightedMarginLoss(loss::MarginLoss, w::Number)
    WeightedMarginLoss(loss, Val(w))
end

@generated function WeightedMarginLoss(s::WeightedMarginLoss{T,W1}, ::Val{W2}) where {T,W1,W2}
    :(WeightedMarginLoss(s.loss, Val($(W1*W2))))
end

for fun in (:value, :deriv, :deriv2)
    @eval function ($fun)(l::WeightedMarginLoss{L,W}, target::Number, output::Number) where {L,W}
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
isclasscalibrated(l::WeightedMarginLoss{T,W}) where {T,W} = W == 0.5 && isclasscalibrated(l.loss)

# TODO: Think about this semantic
issymmetric(::WeightedMarginLoss) = false

for prop in [:isminimizable, :isdifferentiable,
             :istwicedifferentiable,
             :isconvex, :isstrictlyconvex,
             :isstronglyconvex, :isnemitski,
             :isunivfishercons, :isfishercons,
             :islipschitzcont, :islocallylipschitzcont,
             :isclipable, :ismarginbased,
             :isdistancebased]
    @eval ($prop)(l::WeightedMarginLoss) = ($prop)(l.loss)
end

for prop_param in (:isdifferentiable, :istwicedifferentiable)
    @eval ($prop_param)(l::WeightedMarginLoss, at) = ($prop_param)(l.loss, at)
end
