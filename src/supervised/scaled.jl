"""
    ScaledLoss{L,K} <: SupervisedLoss

Can be used to represent a `K` times scaled version of a
given type of loss `L`.
"""
struct ScaledLoss{L<:SupervisedLoss,K} <: SupervisedLoss
    loss::L
end

@generated function (::Type{ScaledLoss{L,K}})(args...) where {L<:SupervisedLoss,K}
    ScaledLoss{L,K}(L(args...))
end
ScaledLoss(loss::T, ::Val{K}) where {T,K} = ScaledLoss{T,K}(loss)
ScaledLoss(loss::T, k::Number) where {T} = ScaledLoss(loss, Val(k))
*(::Val{K}, loss::SupervisedLoss) where {K} = ScaledLoss(loss, Val(K))
*(k::Number, loss::SupervisedLoss) = ScaledLoss(loss, Val(k))

@generated ScaledLoss(s::ScaledLoss{T,K1}, ::Val{K2}) where {T,K1,K2} = :(ScaledLoss(s.loss, Val($(K1*K2))))

for fun in (:value, :deriv, :deriv2)
    @eval @fastmath ($fun)(l::ScaledLoss{T,K}, target::Number, output::Number) where {T,K} = K * ($fun)(l.loss, target, output)
end

for prop in [:isminimizable, :isdifferentiable, :istwicedifferentiable,
             :isconvex, :isstrictlyconvex, :isstronglyconvex,
             :isnemitski, :isunivfishercons, :isfishercons,
             :islipschitzcont, :islocallylipschitzcont,
             :isclipable, :ismarginbased, :isclasscalibrated,
             :isdistancebased, :issymmetric]
    @eval ($prop)(l::ScaledLoss) = ($prop)(l.loss)
end

for prop_param in (:isdifferentiable, :istwicedifferentiable)
    @eval ($prop_param)(l::ScaledLoss, at) = ($prop_param)(l.loss, at)
end
