"""
    OrdinalMarginLoss <: SupervisedLoss

Modifies a margin loss `loss` to be used on an ordinal domain with
number of levels `nlevels`. It treats each level as an integer between
1 and `nlevels`, inclusive, and penalizes output according to the sum of
level thresholds crossed relative to target

Assumes target is encoded in an Index encoding scheme where levels are
numbered between 1 and `nlevels`
"""
struct OrdinalMarginLoss <: SupervisedLoss
    loss::MarginLoss
    nlevels::Int
end

for fun in (:value, :deriv, :deriv2)
    @eval @fastmath function ($fun)(loss::OrdinalMarginLoss, target::Number, output::Number)
        retval = 0
        for t = 1:target - 1
            retval += ($fun)(loss.loss, 1, output - t)
        end
        for t = target + 1:loss.nlevels
            retval += ($fun)(loss.loss, -1, output - t)
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
    @eval function ($fun)(loss::OrdinalMarginLoss, target::Number, output::Number)
        for t = 1:target - 1
            ($fun)(loss.loss, output - t) || return false
        end
        for t = target + 1:loss.nlevels
            ($fun)(loss.loss, t - output) || return false
        end
        return true
    end
end
