
# ==========================================================================
# L(y, t) = ln(1 + exp(-yt))

immutable LogitMarginLoss <: MarginBasedLoss end

value(l::LogitMarginLoss, yt::Number) = log1p(exp(-yt))
deriv(l::LogitMarginLoss, yt::Number) = -1 / (1 + exp(yt))
deriv2(l::LogitMarginLoss, yt::Number) = (eᵗ = exp(yt); eᵗ / abs2(1 + eᵗ))
value_deriv(l::LogitMarginLoss, yt::Number) = (eᵗ = exp(-yt); (log1p(eᵗ), -eᵗ / (1 + eᵗ)))

isunivfishercons(::LogitMarginLoss) = true
isdifferentiable(::LogitMarginLoss) = true
isdifferentiable(::LogitMarginLoss, at) = true
islipschitzcont(::LogitMarginLoss) = true
isconvex(::LogitMarginLoss) = true
isclipable(::LogitMarginLoss) = false

# ==========================================================================
# L(y, t) = max(0, 1 - yt)

immutable L1HingeLoss <: MarginBasedLoss end
typealias HingeLoss L1HingeLoss

value{T<:Number}(l::L1HingeLoss, yt::T) = max(zero(T), 1 - yt)
deriv{T<:Number}(l::L1HingeLoss, yt::T) = yt >= 1 ? zero(T) : -one(T)
deriv2{T<:Number}(l::L1HingeLoss, yt::T) = zero(T)
value_deriv{T<:Number}(l::L1HingeLoss, yt::T) = yt >= 1 ? (zero(T), zero(T)) : (1 - yt, -one(T))

isdifferentiable(::L1HingeLoss) = false
isdifferentiable(::L1HingeLoss, at) = at != 1
islipschitzcont(::L1HingeLoss) = true
isconvex(::L1HingeLoss) = true
isclipable(::L1HingeLoss) = true

# ==========================================================================
# L(y, t) = max(0, 1 - yt)^2

immutable L2HingeLoss <: MarginBasedLoss end

value{T<:Number}(l::L2HingeLoss, yt::T) = yt >= 1 ? zero(T) : abs2(1 - yt)
deriv{T<:Number}(l::L2HingeLoss, yt::T) = yt >= 1 ? zero(T) : 2(yt - one(T))
deriv2{T<:Number}(l::L2HingeLoss, yt::T) = yt >= 1 ? zero(T) : convert(T, -2)
value_deriv{T<:Number}(l::L2HingeLoss, yt::T) = yt >= 1 ? (zero(T), zero(T)) : (abs2(1 - yt), 2(yt - one(T)))

isdifferentiable(::L2HingeLoss) = true
isdifferentiable(::L2HingeLoss, at) = true
islocallylipschitzcont(::L2HingeLoss) = true
isconvex(::L2HingeLoss) = true
isclipable(::L2HingeLoss) = true

# ==========================================================================
# L(y, t) = 0.5 / γ * max(0, 1 - yt)^2   ... yt >= 1 - γ
#           1 - γ / 2 - yt               ... otherwise

immutable L2SmoothedHingeLoss <: MarginBasedLoss
  gamma::Float64

  function L2SmoothedHingeLoss(γ::Number)
    γ > 0 || error("γ must be strictly positive")
    new(convert(Float64, γ))
  end
end

function value{T<:Number}(l::L2SmoothedHingeLoss, yt::T)
  yt >= 1 - l.gamma ? 0.5 / l.gamma * abs2(max(zero(T), 1 - yt)) : one(T) - l.gamma / 2 - yt
end

function deriv{T<:Number}(l::L2SmoothedHingeLoss, yt::T)
  if yt >= 1 - l.gamma
    yt >= 1 ? zero(T) : (yt - one(T)) / l.gamma
  else
    -one(T)
  end
end

function deriv2(l::L2SmoothedHingeLoss, yt::Number)
  yt < 1 - l.gamma && yt >= 1 ? zero(T) : one(T)
end

value_deriv(l::L2SmoothedHingeLoss, yt::Number) = (value(l, yt), deriv(l, yt))

isdifferentiable(::L2SmoothedHingeLoss) = true
isdifferentiable(::L2SmoothedHingeLoss, at) = true
islocallylipschitzcont(::L2SmoothedHingeLoss) = true
isconvex(::L2SmoothedHingeLoss) = true
isclipable(::L2SmoothedHingeLoss) = true
