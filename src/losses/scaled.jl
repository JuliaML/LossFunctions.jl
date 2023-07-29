# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

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
Base.:*(::Val{K}, loss::SupervisedLoss) where {K} = ScaledLoss(loss, Val(K))
Base.:*(k::Number, loss::SupervisedLoss) = ScaledLoss(loss, Val(k))

@generated ScaledLoss(s::ScaledLoss{T,K1}, ::Val{K2}) where {T,K1,K2} = :(ScaledLoss(s.loss, Val($(K1 * K2))))

(l::ScaledLoss{T,K})(output::Number, target::Number) where {T,K} = K * l.loss(output, target)

for FUN in (:deriv, :deriv2)
  @eval function ($FUN)(l::ScaledLoss{T,K}, output::Number, target::Number) where {T,K}
    K * ($FUN)(l.loss, output, target)
  end
end

for FUN in (
  :isminimizable,
  :isdifferentiable,
  :istwicedifferentiable,
  :isconvex,
  :isstrictlyconvex,
  :isstronglyconvex,
  :isnemitski,
  :isunivfishercons,
  :isfishercons,
  :islipschitzcont,
  :islocallylipschitzcont,
  :isclipable,
  :ismarginbased,
  :isclasscalibrated,
  :isdistancebased,
  :issymmetric
)
  @eval ($FUN)(l::ScaledLoss) = ($FUN)(l.loss)
end

for FUN in (:isdifferentiable, :istwicedifferentiable)
  @eval ($FUN)(l::ScaledLoss, at) = ($FUN)(l.loss, at)
end
