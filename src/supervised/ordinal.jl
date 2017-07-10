"""
    OrdinalMarginLoss <: SupervisedLoss

Modifies a margin loss `loss` to be used on an ordinal domain with
number of levels `N`. It treats each level as an integer between
1 and `N`, inclusive, and penalizes output according to the sum of
level thresholds crossed relative to target

Assumes target is encoded in an Index encoding scheme where levels are
numbered between 1 and `N`
"""
struct OrdinalMarginLoss{L<:MarginLoss, N} <: SupervisedLoss
    loss::L
end

function OrdinalMarginLoss(loss::T, ::Type{Val{N}}) where {T<:MarginLoss,N}
  typeof(N) <: Number || _serror()
  OrdinalMarginLoss{T,N}(loss)
end

#=
for fun in (:value, :deriv, :deriv2)
    @eval @fastmath @generated function ($fun)(loss::OrdinalMarginLoss{T, N},
                    target::Number, output::Number) where {T <: MarginLoss, N}
        quote
            retval = zero(output)
            @nexprs $N t -> begin
                not_target = (t != target)
                sgn = sign(target - t)
                retval += not_target * ($($fun))(loss.loss, sgn, output - t)
            end
            retval
        end
    end
end =#

for fun in (:value, :deriv, :deriv2)
    @eval @fastmath function ($fun)(loss::OrdinalMarginLoss{T, N},
                    target::Number, output::Number) where {T <: MarginLoss, N}
        not_target = 1 != target
        sgn = sign(target - 1)
        retval = not_target * ($fun)(loss.loss, sgn, output - 1)
        for t = 2:N
            not_target = (t != target)
            sgn = sign(target - t)
            retval += not_target * ($fun)(loss.loss, sgn, output - t)
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
    @eval function ($fun)(loss::OrdinalMarginLoss{T, N},
                target::Number, output::Number) where {T, N}
        for t = 1:target - 1
            ($fun)(loss.loss, output - t) || return false
        end
        for t = target + 1:N
            ($fun)(loss.loss, t - output) || return false
        end
        return true
    end
end
