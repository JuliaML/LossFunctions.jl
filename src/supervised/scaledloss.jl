immutable ScaledLoss{T<:Number} <: Loss
    loss::Loss
    λ::T
end

value(l::ScaledLoss, r::Number) = l.λ*value(l.loss,r)
deriv(l::ScaledLoss, r::Number) = l.λ*deriv(l.loss,r)
deriv2(l::ScaledLoss, r::Number) = l.λ*deriv2(l.loss,r)
value_deriv(l::ScaledLoss, r::Number) = (l.λ*value(l.loss,r), l.λ*deriv(l.loss,r))

issymmetric(l::ScaledLoss) = issymmetric(l.loss)
isdifferentiable(l::ScaledLoss) = isdifferentiable(l.loss)
isdifferentiable(l::ScaledLoss, at) = isdifferentiable(l.loss, at)
istwicedifferentiable(l::ScaledLoss) = istwicedifferentiable(l.loss)
istwicedifferentiable(l::ScaledLoss, at) = istwicedifferentiable(l.loss, at)
islipschitzcont(l::ScaledLoss) = islipschitzcont(l)
islipschitzcont_deriv(l::ScaledLoss) = islipschitzcont_deriv(l)
isconvex(l::ScaledLoss) = isconvex(l)
isstronglyconvex(l::ScaledLoss) = isstronglyconvex(l)
