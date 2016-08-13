for KIND in (:MarginLoss, :DistanceLoss)
    @eval begin
        immutable $(@compat Symbol("Scaled", KIND)){L<:$KIND,T<:Number} <: $KIND
            loss::L
            factor::T
        end
        (*)(factor::Number, loss::$KIND) = $(Symbol("Scaled", KIND))(loss, factor)
    end
end

typealias ScaledLoss Union{ScaledMarginLoss, ScaledDistanceLoss}

if VERSION >= v"0.5-"
    ScaledLoss(l::Loss, factor::Number) = factor * l
else
    Base.convert(::Type{ScaledLoss}, l::Loss, factor::Number) = factor * l
end

value_deriv(l::ScaledLoss, num::Number) = (l.factor * value(l.loss, num), l.factor * deriv(l.loss, num))

for fun in (:value, :deriv, :deriv2)
    @eval ($fun)(l::ScaledLoss, num::Number) = l.factor .* ($fun)(l.loss, num)
end

for prop in [:isminimizable, :isdifferentiable,
             :istwicedifferentiable, :isconvex,
             :isstronglyconvex, :isnemitski,
             :isunivfishercons, :isfishercons,
             :islipschitzcont, :islocallylipschitzcont,
             :islipschitzcont_deriv, :isclipable,
             :ismarginbased, :isclasscalibrated,
             :isdistancebased, :issymmetric]
    @eval ($prop)(l::ScaledLoss) = ($prop)(l.loss)
end

for prop_param in (:isdifferentiable, :istwicedifferentiable)
    @eval ($prop_param)(l::ScaledLoss, at) = ($prop_param)(l.loss, at)
end

