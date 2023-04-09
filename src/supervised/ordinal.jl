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

for FUN in (:value, :deriv, :deriv2)
    @eval @fastmath function ($FUN)(
            l::OrdinalMarginLoss,
            output::Number,
            target::Number)
        not_target = Int(1 != target)
        sgn = sign(target - 1)
        retval = not_target * ($FUN)(l.loss, output - 1, sgn)
        for t = 2:l.N
            not_target = Int(t != target)
            sgn = sign(target - t)
            retval += not_target * ($FUN)(l.loss, output - t, sgn)
        end
        retval
    end
end

for FUN in [:isminimizable, :isdifferentiable,
             :istwicedifferentiable,
             :isconvex, :isstrictlyconvex,
             :isstronglyconvex, :isnemitski,
             :isunivfishercons, :isfishercons,
             :islipschitzcont, :islocallylipschitzcont]
    @eval ($FUN)(l::OrdinalMarginLoss) = ($FUN)(l.loss)
end