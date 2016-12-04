for KIND in (:MarginLoss, :DistanceLoss)
    SCALEDKIND = Symbol("Scaled", KIND)
    @eval begin
        immutable ($SCALEDKIND){L<:$KIND,K} <: $KIND
            loss::L

            ($SCALEDKIND)(args...) = typeof(K) <: Number ? new(L(args...)) : error()
            ($SCALEDKIND)(loss::L) = typeof(K) <: Number ? new(loss) : error()
        end

        scaledloss{T<:$KIND,K}(loss::T, ::Type{Val{K}}) = ($SCALEDKIND){T,K}(loss)
        ($SCALEDKIND){T}(loss::T, k::Number) = ($SCALEDKIND){T,k}(loss)
        (*)(k::Number, loss::$KIND) = ($SCALEDKIND)(loss, k)
    end
end

typealias ScaledLoss{T,K} Union{ScaledMarginLoss{T,K}, ScaledDistanceLoss{T,K}}

scaledloss(l::Loss, k::Number) = k * l

for fun in (:value, :deriv, :deriv2)
    @eval ($fun){T,K}(l::ScaledLoss{T,K}, num::Number) = K .* ($fun)(l.loss, num)
end

for prop in [:isminimizable, :isdifferentiable,
             :istwicedifferentiable,
             :isconvex, :isstrictlyconvex,
             :isstronglyconvex, :isnemitski,
             :isunivfishercons, :isfishercons,
             :islipschitzcont, :islocallylipschitzcont,
             :isclipable,
             :ismarginbased, :isclasscalibrated,
             :isdistancebased, :issymmetric]
    @eval ($prop)(l::ScaledLoss) = ($prop)(l.loss)
end

for prop_param in (:isdifferentiable, :istwicedifferentiable)
    @eval ($prop_param)(l::ScaledLoss, at) = ($prop_param)(l.loss, at)
end

