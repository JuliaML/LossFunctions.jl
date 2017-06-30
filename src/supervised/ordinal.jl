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
             :islipschitzcont, :islocallylipschitzcont,
             :isclipable, :ismarginbased,
             :isdistancebased]
    @eval ($prop)(l::OrdinalMarginLoss) = ($prop)(l.loss)
end
