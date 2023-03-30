"""
    OrdinalMarginLoss <: SupervisedLoss

Modifies a margin loss `loss` to be used on an ordinal domain with
number of levels `N`. It treats each level as an integer between
1 and `N`, inclusive, and penalizes output according to the sum of
level thresholds crossed relative to target

Assumes target is encoded in an Index encoding scheme where levels are
numbered between 1 and `N`
"""
struct OrdinalMarginLoss{L<:MarginLoss} <: SupervisedLoss
    loss::L
    N::Int
    OrdinalMarginLoss{L}(loss::L, N::Int) where {L<:MarginLoss} = new{L}(loss, N)
end

function OrdinalMarginLoss(loss::L, N::Int) where {L<:MarginLoss}
    OrdinalMarginLoss{L}(loss, N)
end

function (::Type{T})(N::Int, args...) where {T<:OrdinalMarginLoss}
    L = fieldtype(T, 1)
    OrdinalMarginLoss(L(args...), N)
end

for fun in (:value, :deriv, :deriv2)
    @eval @fastmath function ($fun)(
            l::OrdinalMarginLoss,
            target::Number,
            output::Number)
        not_target = Int(1 != target)
        sgn = sign(target - 1)
        retval = not_target * ($fun)(l.loss, sgn, output - 1)
        for t = 2:l.N
            not_target = Int(t != target)
            sgn = sign(target - t)
            retval += not_target * ($fun)(l.loss, sgn, output - t)
        end
        retval
    end
end

for prop in [:isminimizable, :isdifferentiable,
             :istwicedifferentiable,
             :isconvex, :isstrictlyconvex,
             :isstronglyconvex, :isnemitski,
             :isunivfishercons, :isfishercons,
             :islipschitzcont, :islocallylipschitzcont]
    @eval ($prop)(l::OrdinalMarginLoss) = ($prop)(l.loss)
end

for fun in (:isdifferentiable, :istwicedifferentiable)
    @eval function ($fun)(
            l::OrdinalMarginLoss,
            target::Number,
            output::Number)
        for t = 1:target - 1
            ($fun)(l.loss, output - t) || return false
        end
        for t = target + 1:l.N
            ($fun)(l.loss, t - output) || return false
        end
        return true
    end
end
